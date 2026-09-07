"""Validate and explore sparse-memory finite-step response data without a model.

Usage: python -m ngramma_runtime.response_report INPUT_JSON --html OUTPUT_HTML
Only Python's standard library is required; HTML runs offline with no network.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

SCHEMA = 'ngramma.row-response/v1'
QUANTIZER_COUNTS = ('activation_changed_bytes', 'activation_scale_changed_bytes', 'activation_code_changed_bytes')


def _number(value, label, *, nonnegative=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{label} must be a finite number')
    if nonnegative and value < 0:
        raise ValueError(f'{label} must be nonnegative')


def validate_report(report):
    """Validate the public report contract; return the input without mutation."""
    if not isinstance(report, dict) or report.get('schema') != SCHEMA:
        raise ValueError(f'Expected schema {SCHEMA}')
    if report.get('evidence_kind') not in ('synthetic_demo', 'measured'):
        raise ValueError('evidence_kind must be synthetic_demo or measured')
    for owner, key in ((report, 'title'),):
        if not isinstance(owner.get(key), str) or not owner[key].strip():
            raise ValueError(f'{key} must be nonempty text')
    experiment = report.get('experiment')
    if not isinstance(experiment, dict):
        raise ValueError('experiment must describe row_id and direction')
    for key in ('row_id', 'direction'):
        if not isinstance(experiment.get(key), str) or not experiment[key].strip():
            raise ValueError(f'experiment.{key} must be nonempty text')
    interpretation = report.get('interpretation')
    if not isinstance(interpretation, dict) or interpretation.get('response_kind') not in ('finite_step', 'smooth_derivative'):
        raise ValueError('interpretation.response_kind must be finite_step or smooth_derivative')
    for owner in (experiment, interpretation):
        if 'notes' in owner and not isinstance(owner['notes'], str):
            raise ValueError('notes must be text')
    points = report.get('points')
    if not isinstance(points, list) or not 1 <= len(points) <= 10000:
        raise ValueError('points must contain 1–10000 observations')
    seen = set()
    for point in points:
        if not isinstance(point, dict):
            raise ValueError('Every observation must be an object')
        _number(point.get('epsilon'), 'epsilon')
        if point['epsilon'] in seen:
            raise ValueError('Each epsilon must be unique; aggregate repeated trials explicitly')
        seen.add(point['epsilon'])
        _number(point.get('ple_output_rms'), 'ple_output_rms', nonnegative=True)
        for key in ('key_changed_count', 'value_changed_count'):
            if type(point.get(key)) is not int or point[key] < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
            _number(point[key], key)
        for key in QUANTIZER_COUNTS:
            if key in point:
                if type(point[key]) is not int or point[key] < 0:
                    raise ValueError(f'{key} must be a nonnegative integer')
                _number(point[key], key)
        if all(key in point for key in QUANTIZER_COUNTS) and point[QUANTIZER_COUNTS[0]] != point[QUANTIZER_COUNTS[1]] + point[QUANTIZER_COUNTS[2]]:
            raise ValueError('Activation total changed bytes must equal scale plus code changed bytes')
        margin = point.get('engine_logit_margin_delta')
        if margin is not None:
            _number(margin, 'engine_logit_margin_delta')
    baseline = next((point for point in points if point['epsilon'] == 0), None)
    if baseline is None:
        raise ValueError('An epsilon=0 baseline is required')
    if any(baseline[key] != 0 for key in ('ple_output_rms', 'key_changed_count', 'value_changed_count')) or baseline.get('engine_logit_margin_delta') not in (None, 0):
        raise ValueError('The epsilon=0 baseline must have exact zero changes')
    if any(baseline.get(key, 0) != 0 for key in QUANTIZER_COUNTS):
        raise ValueError('The epsilon=0 baseline must have exact zero activation byte changes')
    return report


def _text(value):
    return html.escape(str(value), quote=True)


def _format(value):
    return 'Not supplied' if value is None else format(value, '.7g')


def render_html(report):
    """Render a standalone, accessible data table plus toggleable SVG plots."""
    validate_report(report)
    points = sorted(report['points'], key=lambda row: row['epsilon'])
    demo = report['evidence_kind'] == 'synthetic_demo'
    badge = 'SYNTHETIC DEMO · NOT MODEL EVIDENCE' if demo else 'MEASURED RESPONSE · DOES NOT ESTABLISH IMPROVED CAPABILITY'
    title = ('Synthetic demo: ' if demo else 'Measured response: ') + report['title']
    kind = report['interpretation']['response_kind']
    interpretation = ('Finite-step response: these are changes under explicit perturbations, not a smooth derivative or an admitted gradient.'
                      if kind == 'finite_step' else
                      'Smooth-derivative study: these observations alone do not establish a derivative. Inspect the recorded method and evidence; no gradient qualification is inferred.')
    engine_available = any(point.get('engine_logit_margin_delta') is not None for point in points)
    zero_present = any(point['epsilon'] == 0 for point in points)
    rows = []
    for point in points:
        values = [point['epsilon'], point['ple_output_rms'], point['key_changed_count'],
                  point['value_changed_count'], point.get('engine_logit_margin_delta')]
        rows.append('<tr>'+''.join(f'<td>{_text("Not supplied" if value is None else value)}</td>' for value in values)+'</tr>')
    quantizer_available = any(key in point for point in points for key in QUANTIZER_COUNTS)
    quantizer_table = ''
    if quantizer_available:
        byte_rows = []
        for point in points:
            values = [point['epsilon'], *[point.get(key) for key in QUANTIZER_COUNTS]]
            byte_rows.append('<tr>' + ''.join(f'<td>{_text("Not supplied" if value is None else value)}</td>' for value in values) + '</tr>')
        quantizer_table = '<div class="table-wrap"><table><caption>Native activation quantizer — changed bytes relative to the unedited encoding</caption><thead><tr><th scope="col">Epsilon</th><th scope="col">Total bytes</th><th scope="col">Scale bytes</th><th scope="col">Code bytes</th></tr></thead><tbody>' + ''.join(byte_rows) + '</tbody></table></div><p class="note">Byte counts describe representation changes, not their numerical magnitude. Where all counts are supplied, total = scale + code. Scale-only changes can alter dequantized values even when integer codes are unchanged.</p>'
    panels = [
        ('Native PLE output RMS change', [('ple_output_rms', 'PLE output RMS', '#156b80')]),
        ('Changed key/value elements', [('key_changed_count', 'Key changed count', '#7b47a5'),
                                         ('value_changed_count', 'Value changed count', '#b85116')]),
    ]
    if quantizer_available:
        panels.append(('Native activation quantizer byte changes', [(QUANTIZER_COUNTS[0], 'All changed activation bytes', '#16687b'), (QUANTIZER_COUNTS[1], 'Changed scale bytes', '#915222'), (QUANTIZER_COUNTS[2], 'Changed code bytes', '#5a54aa')]))
    if engine_available:
        panels.append(('Full-engine logit margin change', [('engine_logit_margin_delta', 'Engine margin delta', '#286f47')]))
    # Ordinal epsilon positions preserve visibility across signed dynamic ranges.
    # Normalization by the largest absolute value avoids overflow in subtraction
    # when finite source observations span very large positive/negative values.
    def fraction(value, low, high):
        scale = max(abs(low), abs(high), 1e-300)
        return ((value/scale)-(low/scale))/((high/scale)-(low/scale))
    charts = []
    toggles = []
    for panel_index, (heading, series) in enumerate(panels):
        series = [(key, label, color) for key, label, color in series if any(point.get(key) is not None for point in points)]
        yvalues = [float(point[key]) for key, _, _ in series for point in points if point.get(key) is not None]
        ymin, ymax = min(0.0, min(yvalues)), max(0.0, max(yvalues))
        if ymin == ymax:
            ymax = ymin + 1
        left, top, width, height = 85, 25, 650, 185
        parts = [f'<svg viewBox="0 0 780 260" role="img" aria-labelledby="chart-title-{panel_index}">',
                 f'<title id="chart-title-{panel_index}">{_text(heading)} versus epsilon. Exact observations are in the preceding table.</title>']
        for tick in range(5):
            f = tick/4
            y = top+height*(1-f)
            label = ymin*(1-f)+ymax*f
            parts.append(f'<line x1="{left}" x2="{left+width}" y1="{y}" y2="{y}" class="grid"/><text x="{left-9}" y="{y+4}" text-anchor="end">{_text(_format(label))}</text>')
        tick_indices = list(range(len(points))) if len(points) <= 9 else sorted({round(i*(len(points)-1)/8) for i in range(9)})
        for index in tick_indices:
            x = left + width*(index/max(len(points)-1, 1))
            parts.append(f'<text x="{x}" y="{top+height+24}" text-anchor="middle">{_text(_format(points[index]["epsilon"]))}</text>')
        parts.append('<text x="410" y="251" text-anchor="middle">Signed perturbation ε (evenly spaced observations)</text>')
        for key, label, color in series:
            series_id = 'series-'+key
            toggles.append(f'<label><input type="checkbox" data-series="{series_id}" checked> <span style="color:{color}">{_text(label)}</span></label>')
            parts.append(f'<g id="{series_id}" class="data-series">')
            segment = []
            for index, point in enumerate(points):
                value = point.get(key)
                if value is None:
                    if segment:
                        parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')
                        segment = []
                    continue
                x = left+width*(index/max(len(points)-1, 1))
                y = top+height*(1-fraction(float(value), ymin, ymax))
                segment.append(f'{x:.3f},{y:.3f}')
                parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}"><title>ε={_text(_format(point["epsilon"]))}; {_text(label)}={_text(_format(value))}</title></circle>')
            if segment:
                parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.append('</g>')
        parts.append('</svg>')
        charts.append(f'<section class="chart"><h2>{_text(heading)}</h2>'+''.join(parts)+'</section>')
    notes = ' '.join(str(owner.get('notes', '')) for owner in (report['experiment'], report['interpretation'])).strip()
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; img-src 'none'">
<title>{_text(title)}</title><style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f8;color:#202633;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1040px;margin:auto;padding:35px 22px}}h1{{font-size:clamp(1.7rem,4vw,2.5rem);line-height:1.17;margin:12px 0}}h2{{font-size:1.12rem;margin:0 0 12px}}
.badge{{font-size:.78rem;font-weight:750;letter-spacing:.06em;color:{'#8d4212' if demo else '#225e45'}}}.note{{color:#495265;max-width:85ch}}.facts{{padding:16px 20px;background:#e8edf2;border-radius:8px}}
.table-wrap{{overflow:auto;background:white;border-radius:8px;margin:26px 0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}caption{{text-align:left;padding:16px;font-weight:700}}th,td{{padding:10px 15px;text-align:right;border-bottom:1px solid #e3e7ec;white-space:nowrap}}th{{background:#eaf0f4}}th:first-child,td:first-child{{text-align:left}}
fieldset{{border:1px solid #cbd3dc;padding:12px 15px;border-radius:8px;display:flex;flex-wrap:wrap;gap:10px 22px}}legend{{font-weight:600}}input{{accent-color:#156b80;width:17px;height:17px;vertical-align:middle}}input:focus-visible{{outline:3px solid #156b80;outline-offset:3px}}
.chart{{background:white;padding:20px;margin:18px 0;border-radius:8px}}svg{{display:block;width:100%;height:auto}}svg text{{fill:#465264;font:12px system-ui,sans-serif}}.grid{{stroke:#dce2e8;stroke-width:1}}footer{{font-size:.85rem;color:#596475;margin-top:25px}}
</style></head><body><main><div class="badge">{badge}</div><h1>{_text(title)}</h1>
<p class="note">{_text(interpretation)}</p><div class="facts"><strong>Row:</strong> {_text(report['experiment']['row_id'])}<br>
<strong>Direction:</strong> {_text(report['experiment']['direction'])}<br><strong>Engine evidence:</strong> {'Margin observations supplied for some or all steps.' if engine_available else 'Not supplied; PLE observations do not establish engine behavior.'}<br>
<strong>Zero-step observation:</strong> {'Included.' if zero_present else 'Absent; the reference for the supplied changes must be documented by the producer.'}</div>
<p class="note">{_text(notes)}</p><div class="table-wrap"><table><caption>Recorded observations — primary evidence</caption><thead><tr>
<th scope="col">Epsilon</th><th scope="col">PLE RMS change</th><th scope="col">Key changed</th><th scope="col">Value changed</th><th scope="col">Engine margin Δ</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">PLE change is the absolute RMS difference from the unedited native PLE output, not a relative error. Counts are changed scalar elements across the full sequence in key/value projection outputs. Missing engine values are not zero.</p>
{quantizer_table}
<fieldset><legend>Visible plot series</legend>{''.join(toggles)}</fieldset><p class="note">Connecting lines aid comparison; they do not infer unmeasured responses between steps. Epsilon observations are evenly spaced in sorted order; vertical axes are linear.</p>
{''.join(charts)}<footer>Schema {_text(SCHEMA)} · generated offline by ngramma_runtime.response_report · no model loaded, external assets, or network requests.</footer>
</main><script>
document.querySelectorAll('input[data-series]').forEach(function(input){{
input.addEventListener('change',function(){{document.getElementById(input.dataset.series).style.display=input.checked?'':'none';}});
}});
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_json', type=Path)
    parser.add_argument('--html', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.input_json.read_text(encoding='utf-8'))
        rendered = render_html(report)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.input_json.resolve() == args.html.resolve():
        parser.error('HTML output must differ from the input JSON')
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(rendered, encoding='utf-8')
    print(args.html)


if __name__ == '__main__':
    main()

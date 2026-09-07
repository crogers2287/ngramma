"""Original mock-state verifier, extracted without the task generator."""
import copy

class MockDevices:
    def __init__(self, task):
        self.devices={d['entity_id']:copy.deepcopy(d) for d in task['devices']}
        self.inspected=False;self.errors=[];self.actions=[]

    def call(self,name,args):
        if name=='list_devices':
            if args: return self.error('inventory takes no arguments')
            self.inspected=True
            return {'devices':list(self.devices.values())}
        if name!='set_device':return self.error('unknown tool')
        if set(args)!={'entity_id','state','expected_version'}:return self.error('wrong argument keys')
        if not self.inspected:return self.error('inventory must be inspected before selecting IDs')
        d=self.devices.get(args['entity_id'])
        if d is None:return self.error('invented entity ID')
        if type(args['expected_version']) is not int or args['expected_version']!=d['version']:return self.error('stale or invalid version')
        if args['state'] not in ('on','off'):return self.error('invalid state')
        before=d['state'];d['state']=args['state'];d['version']+=1
        self.actions.append({**args,'before':before})
        return {'ok':True,'device':copy.deepcopy(d)}

    def error(self,message):
        self.errors.append(message)
        return {'error':message}

    def grade(self,task):
        actual={k:d['state'] for k,d in self.devices.items()}
        return {'success':actual==task['expected_state'] and self.inspected and not self.errors,'state_correct':actual==task['expected_state'],'inventory_inspected':self.inspected,'errors':self.errors,'actions':self.actions,'actual_state':actual}

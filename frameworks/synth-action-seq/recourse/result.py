import numpy as np
from recourse.utils import get_instance_info
import time


class SearchResult(object):
    def __init__(self, init_instance):
        self.best_result = None
        self.history = list()
        self.init_instance = init_instance

    def update(self, param_result):
        if param_result is None:
            return
        else:
            self.history.append(param_result)

            if self.best_result is None and param_result.success:
                self.best_result = param_result
            elif self.best_result is not None and param_result.success and param_result.cost < self.best_result.cost:
                self.best_result = param_result



    def __str__(self):
        if self.best_result is None:
            return 'No result'
        else:
            best_result = 'Best Result:\n%s\n' % self.best_result
            return '%s\nOther Results:\n%s' % (best_result, history)

    def summary(self):
        if self.best_result is None:
            return None
        else:
            best_result = self.best_result.summary()
            history = [result.summary() for result in self.history]
            return dict(best_result=best_result,
                        history=history)


class ParamsResult(object):
    def __init__(self, sequence, initial, mode='enumerate'):
        self.name = str(' '.join([action.name for action in sequence]))
        self.loss = float('inf')
        self.f = float('inf')
        self.cost = float('inf')
        self.p = None
        self.sequence = sequence
        self.final_instance = None
        self.final_instance2 = None
        self.mode = mode
        self.initial_instance = initial.reshape((-1,))
        self.final_instance2 = initial

        self.success = False

    def check_precondition(self, p):
        instance = self.initial_instance
        for action in self.sequence:
            new_instance = action.apply(instance, p, use_tensor=False)
            if not action.get_cost(instance, new_instance, use_tensor=False) < np.inf:
                return False
            else:
                instance = new_instance
        return True

    def update(self, loss, cost, f, p, final_instance):

        self.final_instance2 = final_instance

        # print(self.success, f, loss, cost)
        if (self.success and f==0. and cost<self.cost) or (not self.success and f!=0. and f<self.f):
            self.loss = loss
            self.cost = cost
            self.f = f
            self.p = p
            self.final_instance = final_instance
        elif not self.success and f==0.:
            self.success = True
            self.loss = loss
            self.cost = cost
            self.f = f
            self.p = p
            self.final_instance = final_instance


    def __gt__(self, other):
        if self.f == 0 and other.f != 0:
            return True
        elif other.f == 0 and self.f != 0:
            return False
        else:
            return self.cost > other.cost

    def __str__(self):
        sequence = ' '.join([action.name for action in self.sequence])
        if not self.success:
            return 'Sequence <%s> has no Result' % sequence
        else:
            title = 'Sequence <%s>' % sequence
            scores = 'cost=%.9f' % self.cost
            instance_info = dict()
            for action in self.sequence:
                features = action.features
                instance_info = {**instance_info,
                                 **get_instance_info(self.final_instance, features)}
            changed_vals = ['%s=%s' % (k, v) for k, v in instance_info.items()]
            return '\n'.join([title, scores] + changed_vals)

    def summary(self):
        sequence = [action.name for action in self.sequence]
        initial_instance_info = dict()
        final_instance_info = dict()
        for action in self.sequence:
            features = action.features
            initial_instance_info = {**initial_instance_info, **get_instance_info(self.initial_instance, features)}
            final_instance_info = {**final_instance_info, **get_instance_info(self.final_instance, features)}
        p = []
        if self.p is not None:
            p = self.p.tolist()
        return dict(sequence=sequence,
                    params=', '.join([action.print(self.p) for action in self.sequence]),
                    p=p,
                    f=float(self.f),
                    loss=float(self.loss),
                    success=self.success,
                    cost=float(self.cost),
                    initial_instance_info=initial_instance_info,
                    final_instance_info=final_instance_info,
                    final_instance=[] if self.final_instance is None else self.final_instance.tolist())

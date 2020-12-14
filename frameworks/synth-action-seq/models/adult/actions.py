import tensorflow as tf
import numpy as np
from actions.action import Action
import actions.condition_ops as cond


class WaitYears(Action):
    def __init__(self, features):
        super().__init__(name='WaitYears',
                         description='Wait x amount of years',
                         type='Numeric',
                         features=features,
                         target_features=['Age'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['Age'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_age = self.features['Age'].get_feature_value(new_instance, use_tensor,
                                                         space='x')
        old_age = self.features['Age'].get_feature_value(instance, use_tensor, space='x')
        cost = tf.abs(new_age - old_age) if use_tensor else np.abs(new_age - old_age)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_age = self.features['Age'].get_feature_value(new_instance, use_tensor,
                                                         space='x')
        old_age = self.features['Age'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_and(cond.op_gt(new_age, old_age, use_tensor, scale=100.),
                           cond.op_lt(new_age, 120, use_tensor, scale=100.), use_tensor)


class ChangeWorkingHours(Action):
    def __init__(self, features):
        super().__init__(name='ChangeWorkingHours',
                         description='Change number of hours worked per week',
                         type='Numeric',
                         features=features,
                         target_features=['Hours/Week'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['Hours/Week'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_hours = self.features['Hours/Week'].get_feature_value(new_instance, use_tensor,
                                                                  space='x')
        old_hours = self.features['Hours/Week'].get_feature_value(instance, use_tensor,
                                                                  space='x')
        change = (new_hours - old_hours) / 10
        cost = tf.reduce_sum(tf.abs(change)) if use_tensor else np.abs(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_hours = self.features['Hours/Week'].get_feature_value(new_instance, use_tensor,
                                                                  space='x')
        return cond.op_and(cond.op_gt(new_hours, 0, use_tensor, scale=100.),
                           cond.op_lt(new_hours, 90, use_tensor, scale=100.),
                           use_tensor)


class IncreaseCapitalGain(Action):
    def __init__(self, features):
        super().__init__(name='IncreaseCapitalGain',
                         description='Increase Gain in Capital',
                         type='Numeric',
                         features=features,
                         target_features=['Capital Gain'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['Capital Gain'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_gain = self.features['Capital Gain'].get_feature_value(new_instance, use_tensor,
                                                                   space='x')
        old_gain = self.features['Capital Gain'].get_feature_value(instance, use_tensor, space='x')
        change = (new_gain - old_gain) / 10000
        cost = tf.reduce_sum(tf.abs(change)) if use_tensor else np.abs(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def precondition(self, instance, use_tensor=True):
        loss = self.features['Capital Loss'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_lt(loss, 1, use_tensor, scale=10000.)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_gain = self.features['Capital Gain'].get_feature_value(new_instance, use_tensor)
        return cond.op_and(cond.op_gt(new_gain, 0, use_tensor, scale=100000.),
                           cond.op_lt(new_gain, 100000, use_tensor, scale=100000.),
                           use_tensor)


class ChangeCapitalLoss(Action):
    def __init__(self, features):
        super().__init__(name='ChangeCapitalLoss',
                         description='Change Loss in Capital',
                         type='Numeric',
                         features=features,
                         target_features=['Capital Loss'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['Capital Loss'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_loss = self.features['Capital Loss'].get_feature_value(new_instance, use_tensor,
                                                                   space='x')
        old_loss = self.features['Capital Loss'].get_feature_value(instance, use_tensor,
                                                                   space='x')
        change = (new_loss - old_loss) / 1000
        cost = tf.reduce_sum(tf.abs(change)) if use_tensor else np.abs(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def precondition(self, instance, use_tensor=True):
        gain = self.features['Capital Gain'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_lt(gain, 1, use_tensor, scale=10000.)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_loss = self.features['Capital Loss'].get_feature_value(new_instance, use_tensor)
        return cond.op_and(cond.op_gt(new_loss, 1, use_tensor, scale=10000.),
                           cond.op_lt(new_loss, 5000, use_tensor, scale=10000.),
                           use_tensor)


class AddEducation(Action):
    def __init__(self, features):
        super().__init__(name='AddEducation',
                         description='Increase Education Level',
                         type='Numeric',
                         features=features,
                         target_features=['Education num', 'Age'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        change_ed = self.features['Education num'].ztox(self.get_param(p, use_tensor),
                                                        add_mean=False)
        change_age = self.features['Age'].xtoz(change_ed * 2, add_mean=False)
        new_instance = self.features['Education num'].change_feature_value(instance,
                                                                           self.get_param(p,
                                                                                          use_tensor),
                                                                           use_tensor)
        if use_tensor:
            return self.features['Age'].change_feature_value(new_instance, change_age, use_tensor)
        else:
            return self.features['Age'].change_feature_value(new_instance, change_age, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_edu = self.features['Education num'].get_feature_value(new_instance, use_tensor,
                                                                   space='x')
        old_edu = self.features['Education num'].get_feature_value(instance, use_tensor, space='x')
        cost = tf.abs(new_edu - old_edu) if use_tensor else np.abs(new_edu - old_edu)

        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_edu = self.features['Education num'].get_feature_value(new_instance, use_tensor,
                                                                   space='x')
        old_edu = self.features['Education num'].get_feature_value(instance, use_tensor,
                                                                   space='x')
        new_age = self.features['Age'].get_feature_value(new_instance, use_tensor, space='x')
        old_age = self.features['Age'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_and(cond.op_and(cond.op_gt(new_age, old_age, use_tensor, scale=100.),
                                       cond.op_lt(new_age, 120, use_tensor, scale=100.),
                                       use_tensor),
                           cond.op_and(cond.op_gt(new_edu, old_edu, use_tensor, scale=10.),
                                       cond.op_lt(new_edu, 16.5, use_tensor, scale=10.),
                                       use_tensor),
                           use_tensor)


# Categorical Action
class Enlist(Action):
    def __init__(self, features):
        super().__init__(name='Enlist',
                         description='Enlist in Armed Forces',
                         type='Categoric',
                         num_params=0,
                         features=features,
                         target_features=['Occupation'],
                         init_p=[])

    def apply(self, instance, p, use_tensor=True):
        return self.features['Occupation'].change_feature_value(instance, 12, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        return self.return_cost(instance, new_instance, 5., use_tensor)

    def precondition(self, instance, use_tensor=True):
        return cond.op_neq(self.features['Occupation'].get_feature_value(instance, use_tensor),
                           12., use_tensor)


actions = [
    AddEducation,
    ChangeWorkingHours,
    ChangeCapitalLoss,
    Enlist,
    IncreaseCapitalGain,
    WaitYears,
]

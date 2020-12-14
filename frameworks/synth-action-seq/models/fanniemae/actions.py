import tensorflow as tf
import numpy as np

from actions.action import Action
import actions.condition_ops as cond


class ChangeCreditScore(Action):
    def __init__(self, features):
        super().__init__(name='ChangeCreditScore',
                         description='Changes Credit Score',
                         type='Numeric',
                         features=features,
                         target_features=['CreditScore'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['CreditScore'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_credit = self.features['CreditScore'].get_feature_value(new_instance, use_tensor,
                                                                    space='x')
        old_credit = self.features['CreditScore'].get_feature_value(instance, use_tensor,
                                                                    space='x')
        change = (new_credit - old_credit) / old_credit
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_credit = self.features['CreditScore'].get_feature_value(new_instance, use_tensor,
                                                                    space='x')
        return cond.op_and(cond.op_gt(new_credit, 300, use_tensor, scale=1000.),
                           cond.op_lt(new_credit, 850, use_tensor, scale=1000.), use_tensor)


class ChangeNumUnits(Action):
    def __init__(self, features):
        super().__init__(name='ChangeNumUnits',
                         description='Changes NumUnits',
                         type='Numeric',
                         features=features,
                         target_features=['NumUnits'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['NumUnits'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_unit = self.features['NumUnits'].get_feature_value(new_instance, use_tensor, space='x')
        old_unit = self.features['NumUnits'].get_feature_value(instance, use_tensor, space='x')
        change = (new_unit - old_unit) / old_unit
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_unit = self.features['NumUnits'].get_feature_value(new_instance, use_tensor,
                                                               space='x')
        return cond.op_and(cond.op_gt(new_unit, 0, use_tensor, scale=10.),
                           cond.op_lt(new_unit, 5, use_tensor, scale=10.), use_tensor)


class ChangeDTIRat(Action):
    def __init__(self, features):
        super().__init__(name='ChangeDTIRat',
                         description='Changes DTIRat',
                         type='Numeric',
                         features=features,
                         target_features=['DTIRat'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['DTIRat'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_rat = self.features['DTIRat'].get_feature_value(new_instance, use_tensor, space='x')
        old_rat = self.features['DTIRat'].get_feature_value(instance, use_tensor, space='x')
        change = (new_rat - old_rat) / old_rat
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_rat = self.features['DTIRat'].get_feature_value(new_instance, use_tensor,
                                                            space='x')
        return cond.op_and(cond.op_gt(new_rat, 0, use_tensor, scale=100.),
                           cond.op_lt(new_rat, 100, use_tensor, scale=100.), use_tensor)


class ChangeOrInterestRate(Action):
    def __init__(self, features):
        super().__init__(name='ChangeOrInterestRate',
                         description='Changes OrInterestRate',
                         type='Numeric',
                         features=features,
                         target_features=['OrInterestRate'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['OrInterestRate'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_rate = self.features['OrInterestRate'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        old_rate = self.features['OrInterestRate'].get_feature_value(instance, use_tensor,
                                                                     space='x')
        change = (new_rate - old_rate) / old_rate
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_rate = self.features['OrInterestRate'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        return cond.op_and(cond.op_gt(new_rate, 0, use_tensor, scale=100.),
                           cond.op_lt(new_rate, 30, use_tensor, scale=100.), use_tensor)


class ChangeOrLoanTerm(Action):
    def __init__(self, features):
        super().__init__(name='ChangeOrLoanTerm',
                         description='Changes OrLoanTerm',
                         type='Numeric',
                         features=features,
                         target_features=['OrLoanTerm'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['OrLoanTerm'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_term = self.features['OrLoanTerm'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        old_term = self.features['OrLoanTerm'].get_feature_value(instance, use_tensor,
                                                                     space='x')
        change = (new_term - old_term) / old_term
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_term = self.features['OrLoanTerm'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        return cond.op_and(cond.op_gt(new_term, 0, use_tensor, scale=100.),
                           cond.op_lt(new_term, 800, use_tensor, scale=100.), use_tensor)


actions = [
    ChangeCreditScore,
    ChangeNumUnits,
    ChangeDTIRat,
    ChangeOrInterestRate,
    ChangeOrLoanTerm
]

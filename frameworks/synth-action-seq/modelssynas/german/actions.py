import tensorflow as tf
import numpy as np
from actions.action import Action
import actions.condition_ops as cond


# WaitYears
class WaitYears(Action):
    def __init__(self, features):
        super().__init__(name='WaitYears',
                         description='Wait x amount of years',
                         type='Numeric',
                         features=features,
                         target_features=['age_in_years'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['age_in_years'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_age = self.features['age_in_years'].get_feature_value(new_instance, use_tensor,
                                                                  space='x')
        old_age = self.features['age_in_years'].get_feature_value(instance, use_tensor, space='x')
        cost = tf.abs(new_age - old_age) if use_tensor else np.abs(new_age - old_age)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_age = self.features['age_in_years'].get_feature_value(new_instance, use_tensor,
                                                                  space='x')
        old_age = self.features['age_in_years'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_and(cond.op_gt(new_age, old_age, use_tensor, scale=100.),
                           cond.op_lt(new_age, 120, use_tensor, scale=100.), use_tensor)


# Categorical Action
class Naturalize(Action):
    def __init__(self, features):
        super().__init__(name='Naturalize',
                         description='Become citizen',
                         type='Categoric',
                         num_params=0,
                         features=features,
                         target_features=['foreign_worker'],
                         init_p=[])

    def apply(self, instance, p, use_tensor=True):
        return self.features['foreign_worker'].change_feature_value(instance, 1, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        return self.return_cost(instance, new_instance, 5., use_tensor)

    def precondition(self, instance, use_tensor=True):
        return cond.op_neq(self.features['foreign_worker'].get_feature_value(instance, use_tensor),
                           1., use_tensor)


class GetUnskilledJob(Action):
    def __init__(self, features):
        super().__init__(name='GetUnskilledJob',
                         description='Get an unskilled job',
                         type='Categoric',
                         num_params=0,
                         features=features,
                         target_features=['job'],
                         init_p=[])

    def apply(self, instance, p, use_tensor=True):
        return self.features['job'].change_feature_value(instance, 1, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        return self.return_cost(instance, new_instance, 5., use_tensor)

    def precondition(self, instance, use_tensor=True):
        # Must be unemployeed
        return cond.op_and(
            cond.op_and(cond.op_neq(self.features['job'].get_feature_value(instance, use_tensor),
                                    1., use_tensor),
                        cond.op_neq(self.features['job'].get_feature_value(instance, use_tensor),
                                    2., use_tensor),
                        use_tensor),
            cond.op_neq(self.features['job'].get_feature_value(instance, use_tensor),
                        3., use_tensor),
            use_tensor)

        # Categorical Action


class GetGuarantor(Action):
    def __init__(self, features):
        super().__init__(name='GetGuarantor',
                         description='Get a guarantor',
                         type='Categoric',
                         num_params=0,
                         features=features,
                         target_features=['other_debtors_guarantors'],
                         init_p=[])

    def apply(self, instance, p, use_tensor=True):
        return self.features['other_debtors_guarantors'].change_feature_value(instance, 2,
                                                                              use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        return self.return_cost(instance, new_instance, 5., use_tensor)

    def precondition(self, instance, use_tensor=True):
        return cond.op_neq(
            self.features['other_debtors_guarantors'].get_feature_value(instance, use_tensor),
            2., use_tensor)


class ChangeCreditAmount(Action):
    def __init__(self, features):
        super().__init__(name='ChangeCreditAmount',
                         description='Changes Requested Loan Amount',
                         type='Numeric',
                         features=features,
                         target_features=['credit_amount'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['credit_amount'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_credit = self.features['credit_amount'].get_feature_value(new_instance, use_tensor,
                                                                      space='x')
        old_credit = self.features['credit_amount'].get_feature_value(instance, use_tensor,
                                                                      space='x')
        change = (new_credit - old_credit) / old_credit
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def precondition(self, instance, use_tensor=True):
        age = self.features['age_in_years'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_gt(age, 15, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_credit = self.features['credit_amount'].get_feature_value(new_instance, use_tensor,
                                                                      space='x')
        return cond.op_and(cond.op_gt(new_credit, 0, use_tensor, scale=100000.),
                           cond.op_lt(new_credit, 100000, use_tensor, scale=100000.),
                           use_tensor)


class ChangeLoanPeriod(Action):
    def __init__(self, features):
        super().__init__(name='ChangeLoanPeriod',
                         description='Changes Loan Period',
                         type='Numeric',
                         features=features,
                         target_features=['loan_duration'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['loan_duration'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_period = self.features['loan_duration'].get_feature_value(new_instance, use_tensor,
                                                                      space='x')
        old_period = self.features['loan_duration'].get_feature_value(instance, use_tensor,
                                                                      space='x')
        change = (new_period - old_period) / old_period
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_period = self.features['loan_duration'].get_feature_value(new_instance, use_tensor,
                                                                      space='x')
        return cond.op_and(cond.op_gt(new_period, 0, use_tensor, scale=100.),
                           cond.op_lt(new_period, 120, use_tensor, scale=100.),
                           use_tensor)


class AdjustLoanPeriod(Action):
    def __init__(self, features):
        super().__init__(name='AdjustLoanPeriod',
                         description='Changes Loan Period but keeps total loan / period same',
                         type='Numeric',
                         features=features,
                         target_features=['credit_amount', 'loan_duration'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        old_credit_x = self.features['credit_amount'].get_feature_value(instance, use_tensor,
                                                                        space='x')
        old_period_x = self.features['loan_duration'].get_feature_value(instance, use_tensor,
                                                                        space='x')
        change_in_credit_z = self.get_param(p, use_tensor)
        change_in_credit_x = self.features['credit_amount'].ztox(change_in_credit_z, add_mean=False)
        change_in_period_x = old_period_x * (old_credit_x + change_in_credit_x) / old_credit_x
        change_in_period_z = self.features['loan_duration'].xtoz(change_in_period_x, add_mean=False)
        instance = self.features['credit_amount'].change_feature_value(instance, change_in_credit_z,
                                                                       use_tensor)
        return self.features['loan_duration'].change_feature_value(instance, change_in_period_z,
                                                                   use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_credit = self.features['credit_amount'].get_feature_value(new_instance, use_tensor)
        old_credit = self.features['credit_amount'].get_feature_value(instance, use_tensor)
        change = (new_credit - old_credit) / old_credit
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def precondition(self, instance, use_tensor=True):
        old_credit_x = self.features['credit_amount'].get_feature_value(instance, use_tensor,
                                                                        space='x')
        return cond.op_gt(old_credit_x, 1000, use_tensor, scale=100000.)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_period_x = self.features['loan_duration'].get_feature_value(new_instance, use_tensor,
                                                                        space='x')
        new_credit_x = self.features['credit_amount'].get_feature_value(new_instance, use_tensor,
                                                                        space='x')
        return cond.op_and(cond.op_and(cond.op_gt(new_period_x, 0, use_tensor, scale=100.),
                                       cond.op_lt(new_period_x, 120, use_tensor, scale=100.),
                                       use_tensor),
                           cond.op_and(cond.op_gt(new_credit_x, 0, use_tensor, scale=100000.),
                                       cond.op_lt(new_credit_x, 100000, use_tensor, scale=100000.),
                                       use_tensor),
                           use_tensor)


actions = [
    WaitYears,
    Naturalize,
    ChangeCreditAmount,
    ChangeLoanPeriod,
    AdjustLoanPeriod,
    GetGuarantor,
    GetUnskilledJob,
]

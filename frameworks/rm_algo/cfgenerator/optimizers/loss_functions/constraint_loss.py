# NOT BEING USED FOR NOW, NEED FURTHER DEVELOPMENT

def cont_dist_loss(**kwargs):
    return sum((kwargs["x"]-kwargs["cf"])**2)**0.5/kwargs["x"].shape[0]


def d_cont_dist_loss(**kwargs):
    return sum(kwargs["x"]-kwargs["cf"])/(sum((kwargs["x"]-kwargs["cf"]+1e-20)**2)**0.5)/kwargs["x"].shape[0]

def feature_constraint(**kwargs):
    return sum((kwargs["x"]-kwargs["cf"])*[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])**2

def d_feature_constraint(**kwargs):
    return 2*sum((kwargs["x"]-kwargs["cf"])*[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

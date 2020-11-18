import pandas as pd


class Converter:
    def __init__(self, full_df, cat_feats, oh_col):
        self.col = list(full_df.columns)
        self.oh_col = oh_col
        self.oh_dict = {}
        self.cat_feats = cat_feats
        for cat_f in cat_feats:
            values = list(full_df[cat_f].unique())
            for v in values:
                cat_feat_simple = f'{cat_f}_{v}'
                cat_feat_ordered = f'{"".join(["0"]*(4-len(str(cat_f))))}{str(cat_f)}_{v}'
                cat_feat_binary_ordered = f'{"".join(["0"]*(4-len(str(cat_f))))}{str(cat_f)}'

                if cat_feat_simple in oh_col:
                    self.oh_dict[cat_feat_simple] = {cat_f: v}
                elif cat_feat_ordered in oh_col:
                    self.oh_dict[cat_feat_ordered] = {cat_f: v}
                elif cat_feat_binary_ordered in oh_col:
                    self.oh_dict[cat_feat_binary_ordered] = {cat_f: 'binary'}
                else:
                    print('Warning - the key was not found, it will considered a binary')
                    self.oh_dict[cat_f] = {cat_f: 'binary'}

    def convert(self, data):
        if type(data) == pd.DataFrame:
            data = data.to_numpy().tolist()
        output = []
        for d in data:
            out_row = [0]*(len(self.col) - 1)
            for idx, f in enumerate(d):
                of_feat_col = self.oh_col[idx]
                if of_feat_col in self.oh_dict.keys():
                    col_idx, v = list(self.oh_dict[of_feat_col].items())[0]
                    if v == 'binary':
                        out_row[int(col_idx)] = f
                    else:
                        if f == 1:
                            out_row[int(col_idx)] = v
                else:
                    out_row[int(of_feat_col)] = f
            output.append(out_row)

        return output




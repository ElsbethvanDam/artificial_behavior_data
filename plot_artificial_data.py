import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_new_tags(codes):
    beh_codes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 220, 230, 240, 250, 260, 270]
    beh_tags = ['other', 's01_default', 's02_default', 's03_rare', 's04_mixture', 's05_periodic', 's06_highvar_within', 's07_highvar_across', 's08_overlap78', 's09_overlap78', 's10_overlap91011', 's11_periodic_overlap91011', 's12_periodic_overlap91011', 's13_overlap1213', 's14_highvar_overlap1213', 's15_long_overlap1415', 's16_short_overlap1415', 'p20_overlap2021', 'ss21_mix_overlap202123', 'ss22_mix_overlap2223', 'ss23_mix_overlap212223', 'q24_seq_psp_overlap2425', 's25_overlap2425', ]
    # alternative tags
    # beh_tags = ['b00 - trans', 'b01', 'b02', 'x01', 'b05', 'x02', 'x03 - conf 6', 'x04 - conf 6',
    #             'b03 - conf 1', 'b04 - conf 1', 'b10 - conf 3', 'b08 - conf 3', 'b09 - conf 3',
    #             'x05 - conf 7', 'x06 - conf 7', 'b06 - conf 2', 'b07 - conf 2', 'b11 - conf 4',
    #             'b12 - conf 4', 'b13 - conf 4', 'b14 - conf 4', 'b15 - conf 5', 'b16 - conf 5']
    tags = pd.Series(codes).map(dict(zip(beh_codes, beh_tags))) # replace codes with tags
    if tags.isna().any():
        raise ValueError('Could not find tags for class codes: ', [codes[i] for i in np.where(tags.isna())[0]])
    return tags.to_numpy(dtype=object)


class Plotter():
    def __init__(self, class_selection, df_specs, feature_columns):

        beh_codes = df_specs.code_super.values
        beh_tags = df_specs.tag_super.values
        self.lookup = dict(zip(beh_codes, beh_tags))

        classes = df_specs.loc[class_selection].code_super.unique()
        self.all_classes = list(classes)
        self.all_tags = list(pd.Series(classes).map(self.lookup).to_numpy(dtype=object))
        # self.all_tags = get_new_tags(classes) # rename classes if needed

        sns.set_palette("bright", len(classes))
        self.palette = sns.color_palette()

        self.feature_columns = feature_columns


    def plot_pairplot(self, df, out_file=None):

        df['classes'] = list(pd.Series(df['truth_code_super']).map(self.lookup).to_numpy(dtype=object))
        # df['classes'] = get_new_tags(df['truth_code_super']) # rename classes if needed
        columns = self.feature_columns + ['classes']

        # TODO: needed??  df_clf = df[df['truth_code_super'].isin(cfg['clf_classes'])]

        sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
        subset = pd.concat(
            [g.sample(100, replace=(g.count()[0] < 100)) for k, g in df.groupby('classes', sort=True)])
        palette_ids = [self.all_tags.index(c) for c in subset['classes'].unique()]
        if len(self.feature_columns) > 1:
            try:
                g = sns.pairplot(subset.loc[:, columns], hue='classes', palette=[self.palette[c] for c in palette_ids])
            except:
                print("Could not create pairplot (singularities?)")
        else:
            g = sns.displot(subset.loc[:, columns], x='feat1', hue='classes', kind="kde", fill=True, palette=[self.palette[c] for c in palette_ids])
            g.set_axis_labels("feature", "density")
            g.set(yticklabels=[])
        g._legend.set_title("")
        if out_file:
            plt.savefig(out_file)
        else:
            plt.show()


    def plot_timeseries(self, df, with_truth=False, use_truth_sub=False, use_truth_bar=True, with_keys=False, out_file = None):

        plt.clf()
        plt.figure(figsize=[10, 5])
        legend = self.feature_columns.copy()
        columns = self.feature_columns.copy()
        if len(columns) == 1:
            columns = columns[0]
        data = df.loc[:, columns].to_numpy()
        num_features = len(self.feature_columns)
        if num_features > 1:
            for i in range(num_features):
                plt.plot(data[:,i], color=self.palette[-i]) # cycle color backward, so last color is 1 to ensure consistent truth colors between plots
        else:
            plt.plot(data, 'b')

        ax0 = plt.gca()

        if with_truth:

            truth_ids = df['id_sub'].to_numpy() if use_truth_sub else df['id'].to_numpy()
            truth_codes = df['truth_code'].to_numpy() if use_truth_sub else df['truth_code_super'].to_numpy()
            truth_tags = df['truth_tag'].to_numpy() if use_truth_sub else df['truth_tag_super'].to_numpy()

            if use_truth_bar:

                plt.gcf().canvas.draw()  # force creation of ticks
                x_min, x_max = ax0.get_xlim()
                y_min, y_max = ax0.get_ylim()
                # get tick positions relative to axes
                xticks = ax0.get_xticks()
                ticks = [(t - x_min) / (x_max - x_min) for t in ax0.get_xticks()]
                ll = ticks[1]
                ww = ticks[-2] - ll
                n_units = xticks[-2] - xticks[1]
                n_samples = len(truth_ids)
                ww = ww * (n_samples/n_units)
                l, b, w, h = ax0.get_position().bounds
                # calc bb relative to canvas
                ax1 = plt.gcf().add_axes([l + ll*w, .9, ww*w, 0.08*h])

                # _, ids_order = np.unique(ids[s:s+n], return_index=True)
                # palette_ids = [all_tags.index(c) for c in df['classes'][ids_order]]
                sns.heatmap(truth_ids[None,:], cbar=False,
                            xticklabels=False, yticklabels=False,
                            cmap=self.palette,
                            ax=ax1)

                # add sub behavior event log
                # ax2 = plt.gcf().add_axes([l + ll*w, .89, ww*w, 0.03*h])
                # sns.heatmap(subids[None,:], cbar=False,
                #             xticklabels=False, yticklabels=False,
                #             ax=ax2)

            else:
                offset = np.nanmax(data) + .1
                for idx in np.unique(truth_ids, return_index=True)[1]:
                    cl = truth_ids[idx]
                    cl_code = truth_codes[idx]
                    msk = truth_ids == cl
                    if not use_truth_sub:
                        cl_tag = self.all_tags[self.all_classes.index(cl_code)]
                        plt.plot(np.where(msk)[0], offset + truth_ids[msk], 'x', color=self.palette[self.all_classes.index(cl_code)])
                    else:
                        cl_tag = truth_tags[idx]
                        plt.plot(np.where(msk)[0], offset + truth_ids[msk], 'x')
                    legend.append(f'gt_{cl_tag}')
                plt.plot(len(df) + 100, 0, 'b')  # (room for the legend)

        if with_keys:
            msk = df['is_keyframe'].values
            plt.plot(np.where(msk)[0], data[msk], 'o', label='key')
            legend.append('key')

        plt.sca(ax0)
        plt.legend(legend)

        if out_file:
            plt.savefig(out_file)
        else:
            plt.show(block=True)


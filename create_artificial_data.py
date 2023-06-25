import os
import numpy as np
import pandas as pd
import json as js
import scipy.interpolate as si
import markovify as markovify
import logging
import matplotlib.pyplot as plt
import pickle

from plot_artificial_data import Plotter


# sample_types
class sample_types:
    default = 0
    trans = 1
    pose_first = 2
    pose_last = 3
    event_first = 4
    event_last = 5


def get_pose_features(df_specs):
    pose_features = [x[:-4] for x in df_specs.columns if x.endswith('_std')][1:]
    pose_fields = pose_features + ['corr_mean_within', 'corr_periodicity_within_amp', 'corr_periodicity_within_period',
                                   'sample_type']
    return pose_features, pose_fields


def convert_code2ids(codes):
    behaviors = pd.Series(codes).fillna('b99 - unknown').unique() # np.unique(codes)
    onehot = pd.get_dummies(pd.Series(codes)).values
    return np.argmax(onehot, axis=1), behaviors


def get_tags(codes, lookup):
    return pd.Series(codes).map(lookup).to_numpy()  # replace codes with tags


def get_event_ids(ann):
    event_ids = np.zeros_like(ann)
    event_ids[np.diff(ann, prepend=ann[0]) != 0] = 1
    return event_ids.cumsum()


def create_markov_model(transitions_file):

    df_transitions = pd.read_csv(transitions_file, sep=';', index_col='class', skiprows=2).iloc[:, 2:-2]

    text_model = {}
    start_stop_idx = 2
    text_model[tuple(["___BEGIN__"])] = {str(df_transitions.index[start_stop_idx]): 1}
    for i, row in df_transitions.iterrows():
        d = row.to_dict()
        if i == df_transitions.index[start_stop_idx]:
            d["___END__"] = 1
        text_model[tuple([str(i)])] = d
    js_dump = js.dumps(list(text_model.items()))

    # export
    markov_file = transitions_file.replace('class_transitions.csv', 'markov_model.txt')
    with open(markov_file, "w") as f:
        f.write(js_dump)
    print(f'Markov model exported to {markov_file}')


def generate_sequence(markov_model, seq_len=20, class_selection=[]):
    """
    Create sequence of classes according to markov model

    """
    # create sequence
    seq = [markov_model.make_sentence().split(' ')]
    num = len(seq[0])

    # keep adding sentences until long enough
    while num < seq_len:
        # print(num, len(seq), len(seq[-1]))
        last = seq[-1][-1]
        s = markov_model.make_sentence_with_start(last).split(' ')[1:]  # without first element
        if len(s) != 0:
            seq.append(s)
        num += len(s)

    s = [int(float(c)) for sublist in seq for c in sublist]
    s = [x for x in s if x in class_selection][:seq_len]

    # insert missing classes randomly
    for x in [x for x in class_selection if x not in s]:
        s[np.random.randint(0, len(s) - 1)] = x

    return s


def get_nan_poses(n, num_features=1):
    nan_pose = np.full((1, num_features + 4), (np.nan))
    return np.tile(nan_pose, (n, 1))


def get_random_poses(pose_features, specs, n):
    #     print(specs)
    features = []
    for i, ft in enumerate(pose_features):
        feats = np.random.normal(loc=specs[f'{ft}_mean'], scale=specs[f'{ft}_std'], size=n)
        if f'{ft}_min' in specs.index:
            feats[(feats < specs[f'{ft}_min'])] = specs[f'{ft}_min']
        else:
            f_min = specs[f'{ft}_mean'] - 1.5 * specs[f'{ft}_std']
            f_max = specs[f'{ft}_mean'] + 1.5 * specs[f'{ft}_std']
            feats.clip(min=f_min, max=f_max)

        features.append(feats)

    corr_mean_within = np.repeat(specs['corr_mean_within'], n)
    corr_periodicity_within_amp = np.repeat(specs['corr_per_within_amp'], n)
    corr_periodicity_within_per = np.repeat(specs['corr_per_within_period'], n)
    sample_type = np.full(n, sample_types.default)

    return np.vstack(
        (features, corr_mean_within, corr_periodicity_within_amp, corr_periodicity_within_per, sample_type)).T


def get_interpolated_annotation(df_specs, ann, boundaries_to_other=None):
    # replace transitions ('0'-annotation) with nearest non-zero annotation
    # returns annotation of subevents and events
    msk = (ann != 0)
    x = np.arange(len(ann))
    f = si.interp1d(x[msk], ann[msk], kind='nearest', bounds_error=False, fill_value=0)
    ann_sub = f(x)
    ann_super = df_specs.loc[ann_sub, 'code_super'].values.astype(int)
    # insert 00 at boundaries
    if boundaries_to_other is not None:
        if boundaries_to_other != 2:
            raise ('not yet implemented')
        msk_ids = 1 + np.where((np.diff(ann_super, n=2) != 0))[0]
        ann_sub[msk_ids] = 0
        ann_super[msk_ids] = 0
    # plt.figure()
    # plt.plot(ann, 'bx')
    # plt.plot(.2 + ann_sub, 'mx')
    # plt.plot(.6 + ann_super, 'go')
    return ann_sub, ann_super


def get_event_poses(seq, df_specs):
    # function to add extra transition samples
    def add_transition(feats, ann, ev_poses, n=3, num_features=1, behavior=0):
        poses = get_nan_poses(n, num_features)
        poses[:, -1] = sample_types.trans
        feats.append(poses)
        ann.append(np.full(n, behavior))
        ev_poses.append([behavior, n, poses])

    # create data
    pose_features, pose_fields = get_pose_features(df_specs)

    ev_poses = []
    feats = []
    ann = []
    for i, beh in enumerate(seq):

        ev = df_specs.loc[beh]
        #     print(ev)

        num_poses = ev['n_poses']
        if num_poses > 1:
            num_poses = 1 + np.random.choice(num_poses)  # (choice is zero-based)

        if num_poses > 0:
            poses = get_random_poses(pose_features, df_specs.loc[beh], num_poses)
        else:
            poses = get_nan_poses(1, num_features=len(pose_features))
            poses[:, -1] = sample_types.trans

        for ip, p in enumerate(poses):
            dur = max(ev['len_min'], int(np.random.normal(loc=ev['len_mean'], scale=ev['len_std'])))
            ann.append(np.full(dur, beh))
            samples_p = np.tile(p, (dur, 1))
            samples_p[0, -1] = sample_types.pose_first
            samples_p[-1, -1] = sample_types.pose_last
            feats.append(samples_p)
            ev_poses.append([beh, dur, p])
            # set sample_type of first pose sample to start
            if ip == 0:
                feats[-1][0, -1] = sample_types.event_first
            # add transitions
            if ip != len(poses) - 1:
                add_transition(feats, ann, ev_poses, n=(2 + np.random.choice(3)), num_features=len(pose_features),
                               behavior=beh)
            # set sample_type of last pose sample to stop
            if ip == len(poses) - 1:
                feats[-1][-1, -1] = sample_types.event_last

        # add extra transition samples (nan)
        add_transition(feats, ann, ev_poses, n=(3 + np.random.choice(6)), num_features=len(pose_features))

    annotation = np.concatenate(ann)
    features = np.concatenate(feats)

    return ev_poses, features, annotation


# preprocessing filters
def ft_interpolate(y, kind='linear', columns=None):
    if columns is None:
        columns = np.arange(y.shape[1])

    x = np.arange(y.shape[0])
    y_int = y.copy()

    # per column
    for i in columns:
        y_int[:, i] = np.nan
        mask = np.isfinite(y[:, i])
        x_first = x[mask][0]
        x_last = x[mask][-1]
        f = si.interp1d(x[mask], y[mask, i], kind=kind, fill_value="extrapolate")
        y_int[:, i] = f(x)
        y_int[:x_first] = y[:x_first]
        y_int[x_last:] = y[x_last:]

        # import matplotlib.pyplot as plt
        # plt.plot(x, y_int[:,i], '-rx')
        # plt.plot(x, y[:,i], '-bx')

    return y_int


def ft_smoothen(y, window_len=11, columns=None):
    if columns is None:
        columns = np.arange(y.shape[1])
    y_sm = y.copy()
    y_sm[:, columns] = np.nan
    for i in columns:
        y_sm[:, i] = smooth(y[:, i], window_len=window_len)

    return y_sm


# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise (BaseException, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise (BaseException, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (BaseException, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), x, mode='same')
    return y


def ft_add_activity(y, pose_fields, std=None, variation=0.1):
    num_pose_features = len(pose_fields) - 4

    idx_corr_mean = pose_fields.index('corr_mean_within')
    idx_sample_type = pose_fields.index('sample_type')
    corr_mean = y[:, idx_corr_mean]
    corr_per_amp = y[:, pose_fields.index('corr_periodicity_within_amp')]
    corr_per_per = y[:, pose_fields.index('corr_periodicity_within_period')]
    is_periodic = corr_per_per > 0
    sample_type = y[:, idx_sample_type]

    columns = range(num_pose_features)  # all except correlation columns
    activity = np.zeros_like(y)
    periodic_act = np.zeros_like(y)

    num_steps = y.shape[0]
    for j in columns:

        for i in np.arange(num_steps):

            if variation > 0:
                # smoothen activity by dropping samples and interpolate later:
                # The higher the correlation the more samples dropped
                do_keep = np.random.random() < 2 * (1 - corr_mean[i])
                if (sample_type[i] == sample_types.event_first or sample_type[i] == sample_types.event_last or do_keep):
                    if std is None:
                        # activity[i, j] = np.random.normal(loc=0, scale=(max(0, variation * (1 - corr_mean[i]))))
                        activity[i, j] = np.random.normal(loc=0, scale=abs(variation * (1 - corr_mean[i])))
                    else:
                        activity[i, j] = np.random.normal(loc=0, scale=std)
                else:
                    activity[i, j] = np.nan

            # periodicity
            if is_periodic[i]:
                v_amp = 0.1 * np.random.normal(loc=0, scale=variation)
                v_phase = 0.1 * np.random.normal(loc=0, scale=variation)
                # (v*rand(-1,1)+amp)*cos((v*rand(-1,1)+t)*2*pi/period)
                periodic_act[i, j] = (v_amp + corr_per_amp[i]) * np.cos((i + v_phase) * 2 * np.pi / corr_per_per[i])

    activity = ft_interpolate(activity, kind='linear', columns=columns)

    return np.copy(y) + activity + periodic_act


def ft_add_obs_noise(y, pose_fields, std=None):
    idx_corr_mean = pose_fields.index('corr_mean_within')
    noise_obs = np.random.normal(loc=0, scale=1e-3, size=y.shape)
    noise_obs[:, idx_corr_mean:] = 0
    return np.copy(y) + noise_obs


def ft_normalize(y):
    y_norm = np.empty_like(y)
    for i in np.arange(y.shape[1]):
        y_norm[:, i] = np.linalg.norm(y[:, i])
    return y_norm


def ft_constraints(y, pose_constraints):
    y_constr = np.copy(y)
    for i in np.arange(pose_constraints.shape[0]):
        np.clip(y_constr[:, i], pose_constraints[i, 0], pose_constraints[i, 1], out=y_constr[:, i])
    return y_constr


def get_statistics(df_out):
    ann = df_out['truth_code'].values
    df_stats = pd.DataFrame(df_out['truth_tag'].value_counts()).rename(columns={"truth_tag": "nr_frames"})
    df_stats_events = pd.DataFrame(
        df_out.loc[(np.diff(ann, prepend=ann[0] + 1) != 0), 'truth_tag'].value_counts()).rename(
        columns={"truth_tag": "nr_events"})
    df_stats = df_stats.join(df_stats_events)

    df_stats_super = pd.DataFrame(df_out['truth_tag_super'].value_counts()).rename(
        columns={"truth_tag_super": "nr_frames"})
    msk = ~(np.isin(ann, [0]))
    df_out_wo_other = df_out.loc[msk]
    ann_super = df_out['truth_code_super'].values
    df_stats_super_events = pd.DataFrame(df_out_wo_other.loc[(
            np.diff(ann_super[msk], prepend=ann[0] + 1) != 0), 'truth_tag_super'].value_counts()).rename(
        columns={"truth_tag_super": "nr_events"})
    df_stats_super = df_stats_super.join(df_stats_super_events)

    return df_stats, df_stats_super


def create_datasets(specs, specs_dir, out_dir):
    out_dir_base = out_dir

    for cfg in specs:
        dataset_tag, out_affix, ds_subset, num_events, variation, class_selection = cfg

        out_tag = f'{dataset_tag}_{out_affix}_s{num_events}_v{variation}'
        if ds_subset != '':
            dataset_tag_ex = dataset_tag + '_' + ds_subset
            out_tag = f'{out_tag}_{ds_subset}'
        else:
            dataset_tag_ex = dataset_tag

        out_dir = out_dir_base + '/' + out_tag + '/'
        os.makedirs(out_dir, exist_ok=True)

        logging.basicConfig(filename=out_dir + out_tag + '_log.txt',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info(f"Creating artificial data: {dataset_tag_ex}")

        # Read dataset definitions
        df_specs = pd.read_csv(specs_dir + dataset_tag + '_specs.csv', sep=';', index_col='class', comment='#')
        # class;code_super;code_sub;tag_super;tag_sub;n_events;ev_perc;len_mean;len_std;len_min;len_max;len_med;n_poses;pose;feat1_mean;feat1_std;feat1_min;feat1_max;corr_mean_within;corr_per_within_amp;corr_per_within_period
        df_specs['tag_sub'] = df_specs['tag_sub'].fillna('')
        df_specs['tag'] = df_specs['code_super'].astype(str) + ' ' \
                          + df_specs['tag_super'] + ' - ' \
                          + df_specs['code_sub'].astype(str) + ' ' \
                          + df_specs['tag_sub']
        classes = df_specs.index.values

        poses = ['sample', 'trans']
        logging.info(f'classes: {classes}')
        logging.info(f'poses: {poses}')
        logging.info(f'pose_specs: {df_specs.columns}')

        df_specs['code'] = df_specs.index

        beh_codes = df_specs.index.values
        beh_tags = df_specs.tag.values
        lookup = dict(zip(beh_codes, beh_tags))

        if class_selection is None:
            class_selection = beh_codes

        # Get a sequence
        # example ..
        if ds_subset == 'example':
            seq = class_selection
        # .. or generated from text_model
        else:
            # read text_model
            text_model_file = specs_dir + dataset_tag_ex + "_markov_model.txt"
            with open(text_model_file, "r") as f:
                text_model = markovify.Text.from_chain(f.read())
                num_classes = len(text_model.chain.model) - 1
                logging.info(f'MC model loaded. num_classes: {num_classes}')

            seq = generate_sequence(text_model, num_events, class_selection=class_selection)

        num_events = len(seq)
        logging.info(f'num_events: {num_events}')
        poses, feats, ann = get_event_poses(seq, df_specs)
        # print(feats)
        # get nearest annotation for transitions
        nan_feats = np.isnan(feats[:, 0])
        ann, ann_super = get_interpolated_annotation(df_specs, ann, boundaries_to_other=2)

        # Fill in events
        pose_features, pose_fields = get_pose_features(df_specs)
        num_features = len(pose_features)
        feats_int = ft_interpolate(feats, columns=np.arange(num_features + 2))
        feats_constr = ft_constraints(feats_int, np.array([]))  # no constraints for now
        feats_sm = ft_smoothen(feats_constr, window_len=3, columns=np.arange(num_features + 2))
        feats_act = ft_add_activity(feats_sm, pose_fields, variation=variation)
        feats_obs_noise = ft_add_obs_noise(feats_act, pose_fields)
        feats_pp = feats_obs_noise[:, :num_features]
        feats_boundaries = feats[:, -1]
        keys = (feats_boundaries > 1) & (nan_feats == False)
        # feats_nrm = ft_normalize(feats_act)

        # Plot data and annotation (for debugging. Plots to file see below)
        if False:
            n = 1000
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
            ax[0].plot(get_tags(ann[:n], lookup), 'x', label='annotation')
            ax[0].legend()
            if 1: # inspect filter effects
                idx = 0
                ax[1].plot(feats_int[:n, idx], label='interpolated')
                ax[1].plot(feats_constr[:n, idx], label='constraints')
                ax[1].plot(feats_sm[:n, idx], label='smoothed')
                ax[1].plot(feats_act[:n, idx], label='variation')
                # ax[1].plot(feats_nrm[:,idx], label='normalized')
                ax[1].plot(feats_obs_noise[:n, idx], label='observation noise')
                ax[1].plot(feats[:n, idx], label='raw')
                ax[1].set_title(pose_fields[idx])
                ax[1].legend()
            elif 0:
                ax[1].plot(feats_obs_noise)
                ax[1].plot(feats_constr)
                ax[1].plot(feats_act)
                ax[1].plot(feats_sm)
                ax[1].plot(feats_int)
                ax[1].plot(feats)
                ax[1].legend(pose_fields)
            elif 0:
                ax[1].plot(feats_pp[:n])
                ax[1].legend(pose_fields[:num_features])
            else:
                # ax[1].plot(feats[:n], '-.')
                ax[1].plot(feats_pp[:n], '-')
                is_key_t = np.where(feats_boundaries[:n] == 1)[0]  # transition
                is_key_et = np.where(feats_boundaries[:n] == 5)[0]  # event_last
                is_key_es = np.where(feats_boundaries[:n] == 4)[0]  # event_first
                is_key_pt = np.where(feats_boundaries[:n] == 3)[0]  # pose_last
                is_key_ps = np.where(feats_boundaries[:n] == 2)[0]  # pose_first
                ax[1].plot(is_key_t, feats_pp[:n][is_key_t], 'ko', label='trans')
                ax[1].plot(is_key_pt, feats_pp[:n][is_key_pt], 'bo', label='pose_last')
                ax[1].plot(is_key_ps, feats_pp[:n][is_key_ps], 'mo', label='pose_first')
                ax[1].plot(is_key_et, feats_pp[:n][is_key_et], 'go', label='event_last')
                ax[1].plot(is_key_es, feats_pp[:n][is_key_es], 'ro', label='event_first')

            ax[1].set_title('pose features')
            ax[1].legend()
            plt.show()

        # Store everything in dataframes

        # a dataframe with features and labels
        feature_columns = pose_fields[:num_features]
        df_out = pd.DataFrame(feats_pp, columns=feature_columns)
        df_out['truth_code'] = ann
        df_out['truth_code_super'] = ann_super
        df_out['truth_tag'] = df_specs.loc[ann, 'tag'].values
        df_out['truth_tag_super'] = df_specs.loc[ann, 'tag_super'].values
        df_out['event_id'] = get_event_ids(ann)
        df_out['sample_type'] = feats_boundaries
        df_out['is_keyframe'] = keys
        df_out['id_sub'] = convert_code2ids(ann)[0]
        df_out['id'] = convert_code2ids(ann_super)[0]

        # a dataframe with sequence info
        pose_specs = np.stack([p[2] if len(p[2].shape) == 1 else p[2][0] for p in poses])
        df_seq = pd.DataFrame(pose_specs)
        df_seq['code'] = np.stack([p[0] for p in poses])
        df_seq['dur'] = np.stack([p[1] for p in poses])
        df_seq['time'] = df_seq['dur'].cumsum()
        df_seq['event'] = 'S'
        df_seq.loc[(df_seq['dur'] <= 2), 'event'] = 'P'
        df_seq.loc[(df_seq[0].isna()), 'event'] = 'T'
        df_seq['event'] = df_seq['event'] + df_seq['code'].astype(str)

        df_stats, df_stats_super = get_statistics(df_out)

        logging.info('=== sub classes ===')
        logging.info(df_stats)

        logging.info('=== super classes ===')
        logging.info(df_stats_super)

        # Export dataframes
        out_file = out_dir + f'{out_tag}.p'
        pickle.dump(df_out, open(out_file, "wb"))
        logging.info(f"Data saved to {out_file}")
        print(f'Created: {out_file}')

        seq_file = out_dir + f'{out_tag}_seq.csv'
        df_seq.loc[:, ['time', 'event']].to_csv(out_dir + out_tag + '_seq.txt', index=False, sep='\t')
        logging.info(f"Sequence saved to {seq_file}")
        logging.info(f"Tag list:")
        logging.info(df_seq['event'].unique())
        print(f'Created: {seq_file}')

        # Export plots
        plotter = Plotter(class_selection, df_specs, feature_columns)
        plotter.plot_pairplot(df_out, out_file=out_dir + f'{out_tag}_pairplot.png')

        n = min(800, len(df_out))
        s = min(1000, len(df_out)-n)
        plotter.plot_timeseries(df_out[s:s+n], out_file=out_dir + f'{out_tag}_fig.png')
        plotter.plot_timeseries(df_out[s:s+n], with_truth=True, use_truth_sub=False, use_truth_bar=False, out_file=out_dir + f'{out_tag}_fig_truth.png')
        plotter.plot_timeseries(df_out[s:s+n], with_truth=True, use_truth_sub=True, use_truth_bar=False, out_file=out_dir + f'{out_tag}_fig_truth_sub.png')
        n = min(3000, len(df_out))
        plotter.plot_timeseries(df_out[:n], with_truth=True, use_truth_bar=False, out_file=out_dir + f'{out_tag}_fig_truth_{n}.png')

        plt.close('all')

    print('done')


if __name__ == "__main__":

    # Specs
    class_sel_simple = [0, 10, 20, 80]
    class_sel_nostruct = [0, 10, 20, 41, 42, 80, 90, 100, 110, 120, 150, 160]
    class_sel_struct = [0, 10, 20, 220, 231, 232, 241, 242, 243, 244, 251, 252, 261, 262, 263, 270]
    specs = [
        # dataset_tag, out_affix, ds_subset, num_events, variation, classes
        ['ArtifStates_f1', 'c3_simple', 'example', 0, 0.1, class_sel_simple],
        ['ArtifStates_f1', 'c3_1_simple', '', 800, 0.1, class_sel_simple],
        ['ArtifStates_f1', 'c3_2_simple', '', 200, 0.1, class_sel_simple],
        ['ArtifStates_f1', 'c10_nostruct', 'example', 0, 0.5, class_sel_nostruct],
        ['ArtifStates_f1', 'c10_1_nostruct', '', 8000, 0.5, class_sel_nostruct],
        ['ArtifStates_f1', 'c10_2_nostruct', '', 2000, 0.5, class_sel_nostruct],
        ['ArtifStates_f1', 'c8_struct', 'example', 0, 0.5, class_sel_struct],
        ['ArtifStates_f1', 'c8_1_struct', '', 8000, 0.5, class_sel_struct],
        ['ArtifStates_f1', 'c8_2_struct', '', 2000, 0.5, class_sel_struct],
        ['ArtifRat_f4', 'c9', 'example', 0, 0.2, None],
        ['ArtifRat_f4', 'c9_1', '', 8000, 0.2, None],
        ['ArtifRat_f4', 'c9_2', '', 2000, 0.2, None],

    ]

    # convert transition matrices to markov models
    for f in [f for f in os.listdir('specs') if f.endswith('class_transitions.csv')]:
        create_markov_model(r'./specs/' + f)

    # create the data
    create_datasets(specs, specs_dir=r'./specs/', out_dir=r'./generated/')

    print('done')

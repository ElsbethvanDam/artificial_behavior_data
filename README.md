# Disentangling Rodent Behaviors to Improve Automated Behavior Recognition
## Artificial behavior data repository

If you use this code or data, please cite
```
    @article{van17disentangling,
      title={Disentangling Rodent Behaviors to Improve Automated Behavior Recognition},
      author={Van Dam, Elsbeth A and Noldus, Lucas PJJ and Van Gerven, Marcel AJ},
      journal={Frontiers in Neuroscience},
      volume={17},
      pages={1198209},
      publisher={Frontiers}
      doi={10.3389/fnins.2023.1198209}
    }
```

### Introduction
This artificial dataset was designed to develop and evaluate machine learning models that aim to detect the 
behaviors performed by rodents or other agents. What characterizes the behaviors of goal-oriented animals is 
the richness and subtlety of the behavior repertoire, which results in overlap across and variety within poses 
of behavior classes. Also, behaviors can be rare, short or composed of multiple subbehaviors, and are joined 
together by transitions. 

These constituent factors can be configured in this artificial dataset, ranging from 1-dimensional simple state 
behaviors to high dimensional complex patterns.

### Input
As input, the user needs to specify:
- A behavior transition matrix, to specify unbalance and hierarchy 
- Behavior specifications. namely:
  - Event duration (mean, std, min, max, median)
  - Number of subevents per event
  - Feature value distributions (mean, std, min, max)
  - Volatility within an event (smoothness, amount and periodicity)

### Timeseries generation
The procedure to create the timeseries is as follows:
- From the transition matrix, a sequence of events is sampled (via Markov chain + transitions)
- For every event: 
    Number of subevents is sampled
    Then per subevent, we sample duration (clipped to min/max length), and mean and std for every feature
    This is extended with 2 to 8 Nan samples at the end of the events to enable transitions
- Next some temporal filters are applied: interpolation of nans (cubic), smoothening of features
- Finally, for every data point: Feature variation is added based on continuity and periodicity, and observation noise is added with a configurable amount

### Result data
The resulting timeseries is stored as pandas dataframe, for fields see Table ~ref{tab:Datasets}

| Column           | Description |
|------------------|-------------|
| features         | one column per feature |
| truth_code       | unique subbehavior code, numeric |
| truth_code_super | unique behavior code, numeric |
| truth_tag        | string with behavior name (sub class) |
| truth_tag_super  | string with behavior name  |
| event_id         | numeric |
| sample_type      | 0: default, 1: trans, 2 subevent start, 3: subevent last, 4: event start, 5: event last |
| is_key	         | samples with sample type > 1 and valid |


### Generated datasets
	
In this set, 3 different datasets are generated:

| Dataset                      | Description                                                                     |
|------------------------------|---------------------------------------------------------------------------------|
| ArtifStates_f1_c3_simple     | 3 classes that are well separable                                               |
| ArtifStates_f1_c10_nostruct	 | 10 classes without hierarchy                                                    |
| ArtifStates_f1_c8_struct     | 8 classes with hierarchy                                                        |
| ArtifRat_f4_c9               | 9 classes with hierarchy, with feature distributions like in rat behavior |


Every dataset contains 3 timeseries.

File name format: 
`ArtifStates_f[#features]_c[#classes]_[file_id]_[tag]_s[#events]_v[observation noise]`

- 2 files with 8000 resp 2000 events `ArtifStates_[xxx].p`
- 1 file containing 1 event for every class `ArtifStates_[xxx]_example.p`
- And for every generated timeseries, a pairplot of the features and some other plots, namely data w/o truth and sublevel truth.

Note that all event transitions are smooth and that both dataset have an additional 'other'-class for the event transitions.


### Contact
Elsbeth A. van Dam
e.vandam@donders.ru.nl

### Affiliations
* Elsbeth A. van Dam 1,3
* Lucas P.J.J. Noldus 2,3
* Marcel A.J. van Gerven 1

1. Department of Artificial Intelligence, Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, The Netherlands
2. Department of Biophysics, Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, The Netherlands 
3. Noldus Information Technology BV, Wageningen, The Netherlands

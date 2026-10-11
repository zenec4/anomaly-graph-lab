"""Dataset definitions: CSV file names in chronological order and the day-disjoint roles.

Day-disjoint roles:
  cic2018 - train on the first five days, early stopping on 22.02, threshold on 23.02, test on the last three days.
  cic2017 - same chronological logic: train on Monday+Tuesday (the last 15% of their benign windows by time are
            used only for early stopping), threshold on Wednesday, test on Thursday+Friday.
"""
DATASETS = {
    'cic2018': dict(
        days=['Wednesday-14-02-2018.csv', 'Thursday-15-02-2018.csv', 'Friday-16-02-2018.csv', 'Tuesday-20-02-2018.csv',
              'Wednesday-21-02-2018.csv', 'Thursday-22-02-2018.csv', 'Friday-23-02-2018.csv', 'Wednesday-28-02-2018.csv',
              'Thursday-01-03-2018.csv', 'Friday-02-03-2018.csv'],
        roles=dict(FIT=['Wednesday-14-02-2018.csv', 'Thursday-15-02-2018.csv', 'Friday-16-02-2018.csv', 'Tuesday-20-02-2018.csv',
                        'Wednesday-21-02-2018.csv'],
                   CAL=['Thursday-22-02-2018.csv'], VAL=['Friday-23-02-2018.csv'],
                   TEST=['Wednesday-28-02-2018.csv', 'Thursday-01-03-2018.csv', 'Friday-02-03-2018.csv'])),
    'cic2017': dict(
        days=['monday.csv', 'tuesday.csv', 'wednesday.csv', 'thursday.csv', 'friday.csv'],
        roles=dict(FIT=['monday.csv', 'tuesday.csv'], CAL=[], VAL=['wednesday.csv'], TEST=['thursday.csv', 'friday.csv']),
        tail_of_fit=[['CAL', 0.15]], benign_only_days=['monday.csv']),
    # tiny end-to-end check of the scripts (not an experiment)
    'smoke2017': dict(days=['monday.csv', 'thursday.csv'],
                      roles=dict(FIT=['monday.csv'], CAL=[], VAL=[], TEST=['thursday.csv']),
                      tail_of_fit=[['CAL', 0.15], ['VAL', 0.15]], benign_only_days=['monday.csv']),
}

file = '/data/mark/NetworkDatasets/vision_app2/Train/figure_labels.csv'

figure_csv = '/data/mark/NetworkDatasets/vision_app2/Train/' + "figure_labels.csv"
file_csv = open(figure_csv, 'w')
writer = csv.writer(file_csv)
writer.writerow(['Fig_Number', 'Activity_Label'])
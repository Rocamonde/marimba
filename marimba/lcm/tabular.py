

# def get_zero_padded_light_curves(light_curves, num_passbands=6, num_classes=14):
#     data_pairs = list(
#         light_curves
#         [['time', 'flux', 'flux_error', 'band', 'target']]
#         .groupby(['object_id', 'band'])
#         .apply(reset_time)
#         .sample(frac=1)
#         .groupby(['object_id', 'band', 'target'])
#         [['time', 'flux', 'flux_error']]
#     )

#     objects = np.array([pair[0] for pair in data_pairs])
#     data = [np.array(pair[1]) for pair in data_pairs]

#     covariates_list = zero_pad_light_curves(data)
#     covariates = np.array(covariates_list)
#     covariates = covariates.reshape(
#         (int(covariates.shape[0] / num_passbands), num_passbands, covariates.shape[1], covariates.shape[2])).swapaxes(1, 2)

#     classes_mapping, labels_pre1h = np.unique(
#         objects[::num_passbands, 2], return_inverse=True)
#     labels = tf.one_hot(labels_pre1h, num_classes)

#     return covariates, labels

# def reset_time(light_curve):
#     """
#     The following function, to be used on a (grouped) time series, takes a time series dataframe and shifts the time to start at zero.
#     """
#     light_curve['time'] = light_curve['time'] - light_curve['time'].iloc[0]
#     return light_curve

# Implement a first version of multimodal approach


# import needed packages
import time
import keras, numpy, sklearn
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D,\
             Conv2D, MaxPooling2D,\
             Dense, LSTM, GRU, Flatten, Dropout, Reshape, BatchNormalization, concatenate, Input
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import pickle
import seaborn as sns; sns.set()


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def deserialize_dataset(pickle_dataset_filename):
    '''
	Deserialize TC dataset to extract samples and (categorical) labels to perform classification using one of the state-of-the-art DL approaches
	'''

    print('Starting dataset deserialization')

    with open(pickle_dataset_filename, 'rb') as pickle_dataset_file:
      samples = pickle.load(pickle_dataset_file)
      cat_lab_v1 = pickle.load(pickle_dataset_file)
      cat_lab_v2 =  pickle.load(pickle_dataset_file)
      cat_lab_v3 =  pickle.load(pickle_dataset_file)

    input_dim = np.shape(samples)
    num_cls_v1 = numpy.max(cat_lab_v1) + 1
    num_cls_v2 = numpy.max(cat_lab_v2) + 1
    num_cls_v3 = numpy.max(cat_lab_v3) + 1

    print(input_dim)
    print(num_cls_v1)
    print(num_cls_v2)
    print(num_cls_v3)

    print('Ending dataset deserialization')
    return samples, cat_lab_v1, cat_lab_v2, cat_lab_v3

def compute_norm_confusion_matrix(y_true, y_pred, categorical_labels):
    '''
    Compute normalized confusion matrix
    '''

    sorted_labels = sorted(set(categorical_labels))

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=sorted_labels)
    norm_cnf_matrix = preprocessing.normalize(cnf_matrix, axis=1, norm='l1')

    return norm_cnf_matrix

def compute_g_mean(y_true, y_pred):
    '''
    Compute g-mean as the geometric mean of the recalls of all classes
    '''

    recalls = sklearn.metrics.recall_score(y_true, y_pred, average=None)
    nonzero_recalls = recalls[recalls != 0]

    is_zero_recall = False
    unique_y_true = list(set(y_true))

    for i, recall in enumerate(recalls):
        if recall == 0 and i in unique_y_true:
            is_zero_recall = True
            #self.debug_log.error('compute_g_mean: zero-recall obtained, class %s has no sample correcly classified.' % categorical_labels_train[i])

    if is_zero_recall:
        gmean = scipy.stats.mstats.gmean(recalls)
    else:
        gmean = scipy.stats.mstats.gmean(nonzero_recalls)

    return gmean

def compute_topk_accuracy(soft_values, categorical_labels_test, k):
    '''
    Compute top-k accuracy for a given value of k
    '''

    predictions_indices = numpy.argsort(-soft_values, 1)
    predictions_topk_indices = predictions_indices[:,0:k]

    accuracies = numpy.zeros(categorical_labels_test.shape)
    for i in range(k):
        accuracies = accuracies + numpy.equal(predictions_topk_indices[:,i], categorical_labels_test)

    topk_accuracy = sum(accuracies) / categorical_labels_test.size

    return topk_accuracy

def compute_filtered_performance(model, samples_test, soft_values, categorical_labels_test, gamma):
    '''
    Compute filtered performance for a given value of gamma
    '''

    pred_indices = numpy.greater(soft_values.max(1), gamma)
    num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples

    samples_test_filtered = samples_test[numpy.nonzero(pred_indices)]
    categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]

    try:
        filtered_predictions = model.predict_classes(samples_test_filtered, verbose=2)
    except AttributeError:
        filtered_logit_predictions = model.predict(samples_test_filtered)		# Predictions are logit values
        try:
            filtered_predictions = filtered_logit_predictions.argmax(1)
        except AttributeError:
            #self.debug_log.error('Empty list returned by model.predict()')
            sys.stderr.write('Empty list returned by model.predict()\n')
            return numpy.nan, [], numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.array([[numpy.nan] * self.num_classes, [numpy.nan] * self.num_classes]), numpy.nan

    filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
    filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
    filtered_gmean = compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
    filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
    filtered_norm_cnf_matrix = compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions, categorical_labels_test)
    filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape

    return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio

def test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
    '''
    Test the keras model given as input with predict_classes()
    '''

    predictions = model.predict(samples_test, verbose=2).argmax(axis=-1)

    # Calculate soft predictions and test time
    test_time_begin = time.time()
    soft_values = model.predict(samples_test, verbose=2)
    test_time_end = time.time()
    test_time = test_time_end - test_time_begin

    training_predictions = model.predict(samples_train, verbose=2).argmax(axis=-1)

    # print(len(categorical_labels_test))
    # print(categorical_labels_test)

    # print(len(predictions))
    # print(predictions)

    # print(len(soft_values))
    # print(soft_values)

    # Accuracy, F-measure, and g-mean
    accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
    fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
    gmean = compute_g_mean(categorical_labels_test, predictions, categorical_labels_train)
    macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))

    # Accuracy, F-measure, and g-mean on training set
    training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
    training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
    training_gmean = compute_g_mean(categorical_labels_train, training_predictions, categorical_labels_test)
    training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))

    # Confusion matrix
    norm_cnf_matrix = compute_norm_confusion_matrix(categorical_labels_test, predictions, categorical_labels_train)

    # Top-K accuracy
    topk_accuracies = []
    for k in self.K:
        topk_accuracy = compute_topk_accuracy(soft_values, categorical_labels_test, k)
        topk_accuracies.append(topk_accuracy)

    # Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
    filtered_categorical_labels_test_list = []
    filtered_predictions_list = []
    filtered_classified_ratios = []
    filtered_accuracies = []
    filtered_fmeasures = []
    filtered_gmeans = []
    filtered_macro_gmeans = []
    filtered_norm_cnf_matrices = []
    for gamma in self.gamma_range:
        categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
        compute_filtered_performance(model, samples_test, soft_values, categorical_labels_test, gamma)
        filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
        filtered_predictions_list.append(filtered_predictions)
        filtered_accuracies.append(filtered_accuracy)
        filtered_fmeasures.append(filtered_fmeasure)
        filtered_gmeans.append(filtered_gmean)
        filtered_macro_gmeans.append(filtered_macro_gmean)
        filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
        filtered_classified_ratios.append(filtered_classified_ratio)

    return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
    filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, filtered_norm_cnf_matrices, filtered_classified_ratios



samples_w,lab_v1,lab_v2,lab_v3 = deserialize_dataset('../recap-wang/dataset_preparation/576_data_categorical_categorical.pickle')
print(np.shape(samples_w))


num_cla_v1 = np.max(lab_v1) + 1
num_cla_v2 = np.max(lab_v2) + 1
num_cla_v3 = np.max(lab_v3) + 1


samples_w = samples_w[:,0:400]
samples_w = np.expand_dims(samples_w, axis=2)
wang_input_dim  = np.shape(samples_w)[1]
print(np.shape(samples_w))


# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores_eval = []
cvscores_pred = []
fscores_pred  = []

#cvscores_pred_sum = []
#fscores_pred_sum  = []
#
#cvscores_pred_max = []
#fscores_pred_max  = []
#
#cvscores_pred_min = []
#fscores_pred_min  = []
#
#cvscores_pred_mm = []
#fscores_pred_mm  = []

foldNum = 0

# train and test are indices
for train, test in kfold.split(samples_w, lab_v3):

    foldNum = foldNum + 1
    print("Fold no. %d" % (foldNum))

    samples_wan_train = samples_w[train,:]
    samples_wan_test  = samples_w[test,:]

#    samples_lop_train = samples_l[train,:,2:6]
#    samples_lop_test  = samples_l[test,:,2:6]
#    num_fields = 4  #not considering ports and TCP win for Lopez...

    labv1_wan_train  = lab_v1[train]
    labv1_wan_test   = lab_v1[test]

    labv2_wan_train  = lab_v2[train]
    labv2_wan_test   = lab_v2[test]

    labv3_wan_train  = lab_v3[train]
    labv3_wan_test   = lab_v3[test]

#    catlab_lop_train = cat_lab_l[train]
#    catlab_lop_test = cat_lab_l[test]

    ## Cost-sensitive loop for three views
    dic_1 = {}
    dic_2 = {}
    dic_3 = {}

    wt_vec_v1 = np.zeros(num_cla_v1)  # vector of weights for cost-sensitive learning on view1

    for i in range(0,num_cla_v1):
           wt_vec_v1[i] = np.shape(samples_wan_train[labv1_wan_train == i,:])[0]
           dic_1[i] = np.sum(wt_vec_v1) / wt_vec_v1[i]
#    print(wt_vec_v1)
    print(dic_1)

    ## Cost-sensitive loop for three views
    wt_vec_v2 = np.zeros(num_cla_v2)  # vector of weights for cost-sensitive learning on view1
    for i in range(0,num_cla_v2):
           wt_vec_v2[i] = np.shape(samples_wan_train[labv2_wan_train == i,:])[0]
           dic_2[i] = np.sum(wt_vec_v2) / wt_vec_v2[i]
#    print(wt_vec_v2)
    print(dic_2)


    ## Cost-sensitive loop for three views
    wt_vec_v3 = np.zeros(num_cla_v3)  # vector of weights for cost-sensitive learning on view1
    for i in range(0,num_cla_v3):
           wt_vec_v3[i] = np.shape(samples_wan_train[labv3_wan_train == i,:])[0]
           dic_3[i] = np.sum(wt_vec_v3) / wt_vec_v3[i]
#    print(wt_vec_v3)
    print(dic_3)


    class_imbalance_views = { 'VPN': dic_1,
					      'Services': dic_2,
						   'Apps': dic_3}
    print(class_imbalance_views)

    wang_payload = Input(shape = (wang_input_dim,1), name='payload')


    # build 1D-CNN, according to Wang end2end paper
    w =  Conv1D(filters = 32, kernel_size = 25, strides = 1, padding = 'same', activation='relu')(wang_payload)
    w =  MaxPooling1D(pool_size = 3, strides = None, padding = 'same') (w)
    w =  Conv1D(filters = 64, kernel_size = 25, strides = 1, padding = 'same', activation='relu') (w)
    w =  MaxPooling1D(pool_size = 3, strides = None, padding = 'same') (w)
    w =  Flatten() (w)
    w = BatchNormalization() (w)
    w = Dropout(0.2) (w)
    w =  Dense(100, activation = 'relu') (w)


#    y = concatenate([interm_wang, interm_lop])
    w = BatchNormalization() (w)
    w =  Dropout(0.3) (w)
    output_lv1 = Dense(num_cla_v1,  activation = 'softmax', name='VPN')(w)
    output_lv2 = Dense(num_cla_v2,  activation = 'softmax', name='Services')(w)
    output_lv3 = Dense(num_cla_v3,  activation = 'softmax', name = 'Apps')(w)

    multitask_model = Model(inputs=[wang_payload], outputs = [output_lv1,output_lv2,output_lv3])

    multitask_model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'],loss_weights=[0.4, 0.3, 0.3])

    earlystop_multitask = EarlyStopping(monitor='acc', min_delta = 0.01, patience=5, verbose=1, mode='auto')

    callbacks_list_multitask = [earlystop_multitask]

    one_hot_labv1_train = keras.utils.to_categorical(labv1_wan_train, num_cla_v1)
    one_hot_labv2_train = keras.utils.to_categorical(labv2_wan_train, num_cla_v2)
    one_hot_labv3_train = keras.utils.to_categorical(labv3_wan_train, num_cla_v3)

    multitask_model.fit(x = [samples_wan_train], y = [one_hot_labv1_train,one_hot_labv2_train,one_hot_labv3_train], epochs= 50,  batch_size = 50, class_weight=class_imbalance_views, callbacks = callbacks_list_multitask, verbose=2)

    print('Training phase completed')

    [soft_values_multitask_train_v1,soft_values_multitask_train_v2,soft_values_multitask_train_v3] = multitask_model.predict(samples_wan_train,verbose=2)
    multitask_train_pred_v1 = soft_values_multitask_train_v1.argmax(axis=-1)
    multitask_train_pred_v2 = soft_values_multitask_train_v2.argmax(axis=-1)
    multitask_train_pred_v3 = soft_values_multitask_train_v3.argmax(axis=-1)

    accuracy_v1 = sklearn.metrics.accuracy_score(labv1_wan_train, multitask_train_pred_v1)
    fmeas_v1 = sklearn.metrics.f1_score(labv1_wan_train, multitask_train_pred_v1, average='macro')

    accuracy_v2 = sklearn.metrics.accuracy_score(labv2_wan_train, multitask_train_pred_v2)
    fmeas_v2 = sklearn.metrics.f1_score(labv2_wan_train, multitask_train_pred_v2, average='macro')

    accuracy_v3 = sklearn.metrics.accuracy_score(labv3_wan_train, multitask_train_pred_v3)
    fmeas_v3 = sklearn.metrics.f1_score(labv3_wan_train, multitask_train_pred_v3, average='macro')

    test_model(model=multitask_model, samples_train=samples_wan_train, categorical_labels_train=labv1_wan_train, samples_test=samples_wan_test, categorical_labels_test=labv1_wan_test)
    test_model(model=multitask_model, samples_train=samples_wan_train, categorical_labels_train=labv2_wan_train, samples_test=samples_wan_test, categorical_labels_test=labv2_wan_test)
    test_model(model=multitask_model, samples_train=samples_wan_train, categorical_labels_train=labv3_wan_train, samples_test=samples_wan_test, categorical_labels_test=labv3_wan_test)

    print("[Multitask] training_accuracy (v1): %.2f%%" % (accuracy_v1 * 100))
    print("[Multitask] training macro f-measure (v1): %.2f%%" % (fmeas_v1 * 100))

    print("[Multitask] training_accuracy (v2): %.2f%%" % (accuracy_v2 * 100))
    print("[Multitask] training macro f-measure (v2): %.2f%%" % (fmeas_v2 * 100))

    print("[Multitask] training_accuracy (v3): %.2f%%" % (accuracy_v3 * 100))
    print("[Multitask] training macro f-measure (v3): %.2f%%" % (fmeas_v3 * 100))


    # test with predict_classes (multimodal)
    [soft_values_multitask_test_v1,soft_values_multitask_test_v2,soft_values_multitask_test_v3] = multitask_model.predict([samples_wan_test],verbose=2)
    multitask_test_pred_v1 = soft_values_multitask_test_v1.argmax(axis=-1)
    multitask_test_pred_v2 = soft_values_multitask_test_v2.argmax(axis=-1)
    multitask_test_pred_v3 = soft_values_multitask_test_v3.argmax(axis=-1)


    accuracy_mt_v1 = sklearn.metrics.accuracy_score(labv1_wan_test, multitask_test_pred_v1)
    fmeas_mt_v1 = sklearn.metrics.f1_score(labv1_wan_test, multitask_test_pred_v1, average='macro')
    print("[Multitask] predicted_accuracy: %.2f%%" % (accuracy_mt_v1 * 100))
    print("[Multitask] predicted macro f-measure: %.2f%%" % (fmeas_mt_v1 * 100))

    accuracy_mt_v2 = sklearn.metrics.accuracy_score(labv2_wan_test, multitask_test_pred_v2)
    fmeas_mt_v2 = sklearn.metrics.f1_score(labv2_wan_test, multitask_test_pred_v2, average='macro')
    print("[Multitask] predicted_accuracy: %.2f%%" % (accuracy_mt_v2 * 100))
    print("[Multitask] predicted macro f-measure: %.2f%%" % (fmeas_mt_v2 * 100))

    accuracy_mt_v3 = sklearn.metrics.accuracy_score(labv3_wan_test, multitask_test_pred_v3)
    fmeas_mt_v3 = sklearn.metrics.f1_score(labv3_wan_test, multitask_test_pred_v3, average='macro')
    print("[Multitask] predicted_accuracy: %.2f%%" % (accuracy_mt_v3 * 100))
    print("[Multitask] predicted macro f-measure: %.2f%%" % (fmeas_mt_v3 * 100))

    print('test completato')

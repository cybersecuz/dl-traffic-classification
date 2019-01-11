# Implement a first version of multimodal approach


# import needed packages
import sys, os, time, errno, datetime
import keras, numpy, sklearn, imblearn.metrics, scipy
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
from sklearn.metrics import confusion_matrix
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

    input_dim = numpy.shape(samples)
    num_cls_v1 = numpy.max(cat_lab_v1) + 1
    num_cls_v2 = numpy.max(cat_lab_v2) + 1
    num_cls_v3 = numpy.max(cat_lab_v3) + 1

    print("Read %d samples each with %d bytes" % (input_dim[0], input_dim[1]))
    print("Read %d classes for 1st category" % num_cls_v1)
    print("Read %d classes for 2nd category" % num_cls_v2)
    print("Read %d classes for 3rd category" % num_cls_v3)

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

    num_classes = numpy.max(categorical_labels_test) + 1

    try:
        filtered_predictions = model.predict_classes(samples_test_filtered, verbose=2)
    except AttributeError:
        filtered_logit_predictions = model.predict(samples_test_filtered)		# Predictions are logit values
        try:
            filtered_predictions = filtered_logit_predictions.argmax(1)
        except AttributeError:
            #self.debug_log.error('Empty list returned by model.predict()')
            sys.stderr.write('Empty list returned by model.predict()\n')
            return numpy.nan, [], numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.array([[numpy.nan] * num_classes, [numpy.nan] * num_classes]), numpy.nan

    filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
    filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
    filtered_gmean = compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
    filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
    filtered_norm_cnf_matrix = compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions, categorical_labels_test)
    filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape

    return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio

if len(sys.argv) < 2:
	print('Usage: ' + sys.argv[0] + ' <Sample_pickle> <Models dir>')
	sys.exit(1)

samples,lab_v1,lab_v2,lab_v3 = deserialize_dataset(sys.argv[1])
samples = numpy.expand_dims(samples, axis=2)

models_dir = sys.argv[2]


for foldNum in range(1,11):
    
    with open('%s/%d.pickle' % (models_dir, foldNum), 'rb') as pickle_dataset_file:
        samples_wan_train_indices = pickle.load(pickle_dataset_file)
        samples_wan_test_indices = pickle.load(pickle_dataset_file)
    
    samples_wan_train = samples[samples_wan_train_indices,:]
    samples_wan_test  = samples[samples_wan_test_indices,:]
    
    labv1_wan_train  = lab_v1[samples_wan_train_indices]
    labv1_wan_test   = lab_v1[samples_wan_test_indices]

    labv2_wan_train  = lab_v2[samples_wan_train_indices]
    labv2_wan_test   = lab_v2[samples_wan_test_indices]

    labv3_wan_train  = lab_v3[samples_wan_train_indices]
    labv3_wan_test   = lab_v3[samples_wan_test_indices]

    multitask_model = keras.models.load_model(models_dir + "/%d.h5" % (foldNum))

    print("Loaded model %d" % (foldNum))

    [soft_values_multitask_train_v1,soft_values_multitask_train_v2,soft_values_multitask_train_v3] = multitask_model.predict(samples_wan_train,verbose=2)
    multitask_train_pred_v1 = soft_values_multitask_train_v1.argmax(axis=-1)
    multitask_train_pred_v2 = soft_values_multitask_train_v2.argmax(axis=-1)
    multitask_train_pred_v3 = soft_values_multitask_train_v3.argmax(axis=-1)

    accuracy_v1 = sklearn.metrics.accuracy_score(labv1_wan_train, multitask_train_pred_v1)
    fmeas_v1 = sklearn.metrics.f1_score(labv1_wan_train, multitask_train_pred_v1, average='macro')
    gmean_v1 = compute_g_mean(labv1_wan_train, multitask_train_pred_v1)

    accuracy_v2 = sklearn.metrics.accuracy_score(labv2_wan_train, multitask_train_pred_v2)
    fmeas_v2 = sklearn.metrics.f1_score(labv2_wan_train, multitask_train_pred_v2, average='macro')
    gmean_v2 = compute_g_mean(labv2_wan_train, multitask_train_pred_v2)

    accuracy_v3 = sklearn.metrics.accuracy_score(labv3_wan_train, multitask_train_pred_v3)
    fmeas_v3 = sklearn.metrics.f1_score(labv3_wan_train, multitask_train_pred_v3, average='macro')
    gmean_v3 = compute_g_mean(labv3_wan_train, multitask_train_pred_v3)

    print("[Multitask] training_accuracy (v1): %.2f%%" % (accuracy_v1 * 100))
    print("[Multitask] training macro f-measure (v1): %.2f%%" % (fmeas_v1 * 100))
    print("[Multitask] training g-mean (v1): %.2f%%" % (gmean_v1 * 100))

    print("[Multitask] training_accuracy (v2): %.2f%%" % (accuracy_v2 * 100))
    print("[Multitask] training macro f-measure (v2): %.2f%%" % (fmeas_v2 * 100))
    print("[Multitask] training g-mean (v2): %.2f%%" % (gmean_v2 * 100))

    print("[Multitask] training_accuracy (v3): %.2f%%" % (accuracy_v3 * 100))
    print("[Multitask] training macro f-measure (v3): %.2f%%" % (fmeas_v3 * 100))
    print("[Multitask] training g-mean (v3): %.2f%%" % (gmean_v3 * 100))


    # test with predict_classes (multimodal)
    [soft_values_multitask_test_v1,soft_values_multitask_test_v2,soft_values_multitask_test_v3] = multitask_model.predict([samples_wan_test],verbose=2)
    multitask_test_pred_v1 = soft_values_multitask_test_v1.argmax(axis=-1)
    multitask_test_pred_v2 = soft_values_multitask_test_v2.argmax(axis=-1)
    multitask_test_pred_v3 = soft_values_multitask_test_v3.argmax(axis=-1)


    accuracy_mt_v1 = sklearn.metrics.accuracy_score(labv1_wan_test, multitask_test_pred_v1)
    fmeas_mt_v1 = sklearn.metrics.f1_score(labv1_wan_test, multitask_test_pred_v1, average='macro')
    print("[Multitask] predicted_accuracy: %.2f%%" % (accuracy_mt_v1 * 100))
    print("[Multitask] predicted macro f-measure: %.2f%%" % (fmeas_mt_v1 * 100))
    print(".:Confusion Matrix v1")
    norm_conf_matriv_v1 = compute_norm_confusion_matrix(labv1_wan_test, multitask_test_pred_v1, labv1_wan_test)
    print(norm_conf_matriv_v1)

    accuracy_mt_v2 = sklearn.metrics.accuracy_score(labv2_wan_test, multitask_test_pred_v2)
    fmeas_mt_v2 = sklearn.metrics.f1_score(labv2_wan_test, multitask_test_pred_v2, average='macro')
    top3_accouracy_v2 = compute_topk_accuracy(soft_values_multitask_test_v2, labv2_wan_test, 3)
    print("[Multitask] predicted_accuracy: %.2f%%" % (accuracy_mt_v2 * 100))
    print("[Multitask] predicted macro f-measure: %.2f%%" % (fmeas_mt_v2 * 100))
    print("[Multitask] predicted macro top3 accuracy: %.2f%%" % (top3_accouracy_v2 * 100))
    print(".:Confusion Matrix v2")
    norm_conf_matriv_v2 = compute_norm_confusion_matrix(labv2_wan_test, multitask_test_pred_v2, labv2_wan_test)
    print(norm_conf_matriv_v2)

    accuracy_mt_v3 = sklearn.metrics.accuracy_score(labv3_wan_test, multitask_test_pred_v3)
    fmeas_mt_v3 = sklearn.metrics.f1_score(labv3_wan_test, multitask_test_pred_v3, average='macro')
    top3_accouracy_v3 = compute_topk_accuracy(soft_values_multitask_test_v3, labv3_wan_test, 3)
    print("[Multitask] predicted_accuracy: %.2f%%" % (accuracy_mt_v3 * 100))
    print("[Multitask] predicted macro f-measure: %.2f%%" % (fmeas_mt_v3 * 100))
    print("[Multitask] predicted macro top3 accuracy: %.2f%%" % (top3_accouracy_v3 * 100))
    print(".:Confusion Matrix v3")
    norm_conf_matriv_v3 = compute_norm_confusion_matrix(labv3_wan_test, multitask_test_pred_v3, labv3_wan_test)
    print(norm_conf_matriv_v3)

    print('test completato')

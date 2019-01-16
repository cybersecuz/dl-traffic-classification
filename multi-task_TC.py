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

if len(sys.argv) < 3:
	print('Usage: ' + sys.argv[0] + ' <Categorical data file> <Models dir>')
	sys.exit(1)

samples_w,lab_v1,lab_v2,lab_v3 = deserialize_dataset(sys.argv[1])

num_cla_v1 = numpy.max(lab_v1) + 1
num_cla_v2 = numpy.max(lab_v2) + 1
num_cla_v3 = numpy.max(lab_v3) + 1

# taglia a 400 i byte scelti
# samples_w = samples_w[:,0:400]

samples_w = numpy.expand_dims(samples_w, axis=2)
wang_input_dim = numpy.shape(samples_w)[1]
print(numpy.shape(samples_w))
print(wang_input_dim)


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

    labv1_wan_train  = lab_v1[train]
    labv1_wan_test   = lab_v1[test]

    labv2_wan_train  = lab_v2[train]
    labv2_wan_test   = lab_v2[test]

    labv3_wan_train  = lab_v3[train]
    labv3_wan_test   = lab_v3[test]

    dataset_filename = '%s/%d.pickle' % (sys.argv[2], foldNum)
    
    if os.path.isfile(dataset_filename):
        os.remove(dataset_filename)
    
    with open(dataset_filename, 'wb') as dataset_file:
        pickle.dump(train, dataset_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test, dataset_file, pickle.HIGHEST_PROTOCOL)


# Ora sta bilanciando con dei pesi le varie label nel caso ci fossero troppi dati di una label rispetto ad un'altra
    ## Cost-sensitive loop for three views
    dic_1 = {}
    dic_2 = {}
    dic_3 = {}

    wt_vec_v1 = numpy.zeros(num_cla_v1)  # vector of weights for cost-sensitive learning on view1

    for i in range(0,num_cla_v1):
           wt_vec_v1[i] = numpy.shape(samples_wan_train[labv1_wan_train == i,:])[0]
           dic_1[i] = numpy.sum(wt_vec_v1) / wt_vec_v1[i]
#    print(wt_vec_v1)

    ## Cost-sensitive loop for three views
    wt_vec_v2 = numpy.zeros(num_cla_v2)  # vector of weights for cost-sensitive learning on view1
    for i in range(0,num_cla_v2):
           wt_vec_v2[i] = numpy.shape(samples_wan_train[labv2_wan_train == i,:])[0]
           dic_2[i] = numpy.sum(wt_vec_v2) / wt_vec_v2[i]
#    print(wt_vec_v2)


    ## Cost-sensitive loop for three views
    wt_vec_v3 = numpy.zeros(num_cla_v3)  # vector of weights for cost-sensitive learning on view1
    for i in range(0,num_cla_v3):
           wt_vec_v3[i] = numpy.shape(samples_wan_train[labv3_wan_train == i,:])[0]
           dic_3[i] = numpy.sum(wt_vec_v3) / wt_vec_v3[i]
#    print(wt_vec_v3)


    class_imbalance_views = {'VPN': dic_1, 'Services': dic_2, 'Apps': dic_3}
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

    multitask_model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'], loss_weights=[0.4, 0.3, 0.3])

    earlystop_multitask_VPN_acc = EarlyStopping(monitor='VPN_acc', min_delta = 0.01, patience=5, verbose=1, mode='auto')
    earlystop_multitask_Services_acc = EarlyStopping(monitor='Services_acc', min_delta = 0.01, patience=5, verbose=1, mode='auto')
    earlystop_multitask_Apps_acc = EarlyStopping(monitor='Apps_acc', min_delta = 0.01, patience=5, verbose=1, mode='auto')

    callbacks_list_multitask = [earlystop_multitask_VPN_acc, earlystop_multitask_Services_acc, earlystop_multitask_Apps_acc]

    one_hot_labv1_train = keras.utils.to_categorical(labv1_wan_train, num_cla_v1)
    one_hot_labv2_train = keras.utils.to_categorical(labv2_wan_train, num_cla_v2)
    one_hot_labv3_train = keras.utils.to_categorical(labv3_wan_train, num_cla_v3)

    multitask_model.fit(x = [samples_wan_train], y = [one_hot_labv1_train,one_hot_labv2_train,one_hot_labv3_train], epochs= 50,  batch_size = 50, class_weight=class_imbalance_views, callbacks = callbacks_list_multitask, verbose=2)
    print('Training phase completed')
    
    multitask_model.save(sys.argv[2] + "/%d.h5" % (foldNum))
    print('Models saved')
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.externals import joblib
from subprocess import call
from PIL import Image
from util import check_N_mkdir, _tifReader
from itertools import repeat
from filter import *


def train(X, y, params=None):
    clf = RandomForestClassifier(
        n_estimators=params['nb_tree'],  # 200
        max_depth=params['depth'],  # None: infinite depth
        n_jobs=-1,  # -1 for all CPU
        max_features=None,  #“auto”: max_features=sqrt(n_features)
        verbose=1,
    )
    clf.fit(X, y)
    n_nodes = []
    max_depths = []

    for ind_tree in clf.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)
    print('Average number of nodes {}'.format(int(np.mean(n_nodes))))
    print('Average maximum depth {}'.format(int(np.mean(max_depths))))
    return clf


def train_auto_optimized_model(X, y, param_grid=None):
    clf = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,
                            scoring='roc_auc', cv=1, verbose=1)
    clf.fit(X, y)
    return clf


def save_model(model, model_dir=None):
    check_N_mkdir(model_dir)
    joblib.dump(model, model_dir + 'model.sav')


def load_model(model_path=None):
    model = joblib.load(model_path)
    return model


def test(X, y, model):
    score = model.score(X, y)
    return score


def predict(stackImg, model, rlt_dir=None, filt_names=None):
    assert isinstance(stackImg, list), 'stackImg should be a list of image for ignoring different input images size'
    check_N_mkdir(rlt_dir)
    shapes = [img.shape for img in stackImg]
    sizes = [img.size for img in stackImg]

    # compute featuremaps
    inf_features = get_featureMaps(stackImg, filter_names=filt_names)
    inf_input = make_input(inf_features)

    # predict
    rlt = model.predict(inf_input)

    # reshape result to list of img
    pointer = 0
    i = 0
    out = []
    for shp, siz in zip(shapes, sizes):
        _rlt = rlt[pointer: pointer + siz].reshape(shp)
        Image.fromarray(_rlt).save(rlt_dir + '{}.tif'.format(i))
        out.append(_rlt)
        pointer += siz
        i += 1
    return out


def export_treeGraph(model, l_filt=None, graph_dir=None):
    check_N_mkdir(graph_dir)
    export_graphviz(model, graph_dir + 'graph.dot', rounded=True,
                    feature_names=l_filt, max_depth=8,
                    class_names=['NMC', 'CBD', 'pore'], filled=True)
    # should apt graphviz
    try:
        call(['dot', '-Tpng', graph_dir + 'graph.dot', '-O', ''.format(graph_dir + 'graph.png'), '-Gdpi=50'])
    except Exception as e:
        print('Error msg: {}'.format(e))
        call(['dot', '-Tpng', graph_dir + 'graph.dot', '-O', ''.format(graph_dir + 'graph.png'), '-Gdpi=40'])


def get_featureMaps(stackImg, filter_names=None):
    '''

    :param stackImg: (list) list of 2D np array for image
    :param filter_names: (list)
    :return:
    '''
    assert isinstance(stackImg, list), 'stackImg should be a list'
    assert isinstance(filter_names, list), 'filter_names should be a list'
    # inits
    dic_filter = {
        'gaussian_blur': Gaussian_Blur,
        'sobel': Sobel,
        'hessian': Hessian,
        'dog': DoG,
        'gabor': Gabor,
        # 'membrane_proj': Membrane_proj,
        'anisotropic_diffusion1': Anisotropic_Diffusion1,
        'anisotropic_diffusion2': Anisotropic_Diffusion2,
        'bilateral': Bilateral,
        'median': Median,
    }

    # lowercase
    filter_names = [i.replace(' ', '_').lower() for i in filter_names]
    l_func = []
    featureMaps = []
    for filt_n in filter_names:
        l_func.append(dic_filter[filt_n])

    pool = mp.Pool(processes=mp.cpu_count())
    for img in stackImg:
        # if chunksize = 1, results is in order
        _res = pool.starmap(wrapper, zip(l_func, repeat(img)), chunksize=1)
        featureMaps.append(_res)
    return featureMaps


def get_data(stackFeatures, stackLabels):
    ''''''
    # init ndarray of the data
    nb_filt = len(stackFeatures[0])
    total_pixel = 0
    for i, feat in enumerate(stackFeatures):
        total_pixel += feat[0].size
    data = np.empty((total_pixel, nb_filt + 1))
    _data = np.empty(total_pixel)

    # fill data
    for i in range(nb_filt):
        place = 0
        for j, feat in enumerate(stackFeatures):
            _data[place: place + feat[i].size] = feat[i].flatten()
            place += feat[i].size
        data[:, i] = _data

    # put labels
    place = 0
    for label in stackLabels:
        data[place: place + label.size, -1] = label.flatten()
        place += label.size

    return data[:, :-1], data[:, -1]


def make_input(stackFeatures):
    ''''''
    # init ndarray of the data
    nb_filt = len(stackFeatures[0])
    total_pixel = 0
    for i, feat in enumerate(stackFeatures):
        total_pixel += feat[0].size
    data = np.empty((total_pixel, nb_filt))
    _data = np.empty(total_pixel)

    # fill data
    for i in range(nb_filt):
        place = 0
        for j, feat in enumerate(stackFeatures):
            _data[place: place + feat[i].size] = feat[i].flatten()
            place += feat[i].size
        data[:, i] = _data

    return data


if __name__ == '__main__':
    # inits
    paths = {
        'raw_dir': './raw/',
        'rlt_dir': './RF_result/rlt/',
        'model_dir': './RF_result/mdl/'
    }

    filt_names = [
        'gaussian_blur',
        'sobel',
        'hessian',
        'dog',
        #'membrane_proj',
        'anisotropic_diffusion1',
        'anisotropic_diffusion2',
        'gabor',
        'bilateral',
        'median',
    ]

    params = {
        'nb_tree': 200,
        'depth': 20,
    }

    test_grid = {
        'n_estimators': np.linspace(10, 200).astype(int),
        'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
        'max_features': ['auto', None] + list(np.arange(0.5, 1, 0.1)),
        'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    }

    # prepare
    X_stack, y_stack, _ = _tifReader(paths['raw_dir'])
    stackFeatures = get_featureMaps(X_stack, filt_names)
    X, y = get_data(stackFeatures, y_stack)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # train
    clf = train(X_train, y_train, params=params)
    # clf = train_auto_optimized_model(X_train, y_train, param_grid=test_grid)

    # test
    # todo: should verify the testset
    score = test(X_test, y_test, clf)
    print(score)
    export_treeGraph(clf.estimators_[0], l_filt=filt_names, graph_dir=paths['model_dir'])
    save_model(clf, model_dir=paths['model_dir'])
    clf = load_model(model_path=paths['model_dir'] + 'model.sav')

    # produce
    inference_stack, _, _ = _tifReader(paths['raw_dir'])
    vol_RF = predict(inference_stack, clf, rlt_dir=paths['rlt_dir'], filt_names=filt_names)


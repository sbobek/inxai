from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import shap
import lime


def create_intermediate_points(start_vals, end_vals, resolution):
    arr = []
    for start_val, end_val in zip(start_vals, end_vals):
        arr.append(np.linspace(start_val, end_val, resolution))
    return np.array(arr).T


def generate_per_instance_importances(models, X, y, framework='tree_shap'):
    """
    It generates explanations per insance using predefined framework.
    It is wise to subsample training set, as calculating explanations is time consuming
    especially for frameworks such as LIME.

    :param models:
    :param X:
    :param y:
    :param framework:
    :return:
    """
    importances_per_model = []
    if type(models) != list:
        models = [models]
    for model in models:
        if framework == 'tree_shap':
            explainer = shap.TreeExplainer(model)
            all_importances = explainer.shap_values(X)

            # If is multiclass, choose explanation for the correct class
            if isinstance(all_importances, list):
                right_imps = []
                for idx, label in enumerate(y):
                    right_imps.append(all_importances[label][idx])
                all_importances = right_imps
        elif framework == 'kernel_shap':
            explainer = shap.KernelExplainer(model.predict_proba, X)
            all_importances = explainer.shap_values(X)

            # If is multiclass, choose explanation for the correct class
            if isinstance(all_importances, list):
                right_imps = []
                for idx, label in enumerate(y):
                    right_imps.append(all_importances[label][idx])
                all_importances = right_imps
        elif framework == 'lime':
            all_importances = []

            explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns)

            for index, (skip, row) in enumerate(X.iterrows()):
                correct_label = y[index]

                # If is multiclass, choose explanation for the correct class
                exp = explainer.explain_instance(row, model.predict_proba, num_features=len(X.columns),
                                                 labels=(correct_label,))
                imps = dict()

                for feat in exp.local_exp[correct_label]:
                    imps[feat[0]] = feat[1]
                imp_vals = []
                for i in range(len(imps)):
                    imp_vals.append(imps[i])

                all_importances.append(imp_vals)

        else:
            print('Bad framework.')
            return None
        importances_per_model.append(all_importances)

    if len(importances_per_model) == 1:
        return importances_per_model[0]
    else:
        return importances_per_model


class GlobalFeatureMetric:
    """

    """

    def gradual_perturbation(self, model, X, y, importances_orig, column_transformer, preprocessing_pipeline=None,
                             resolution=10, count_per_step=5, plot=True):
        """


        :param model:
        :param X:
        :param y:
        :param importances_orig:
        :param column_transformer:
        :param preprocessing_pipeline:
        :param resolution:
        :param count_per_step:
        :return:
        """

        baseline_predictions = model.predict(X)
        baseline_accuracy = accuracy_score(y, baseline_predictions)
        inv_norm_importances = 1 - abs(importances_orig) / (sum(abs(importances_orig)))

        intermediate_importances = create_intermediate_points(np.zeros(len(inv_norm_importances)),
                                                              inv_norm_importances, resolution)

        accuracies = []
        for importances in intermediate_importances:
            this_step_accuracies = self.gradual_perturbation_step(model=model, X=X, y=y,
                                                                  importances=importances,
                                                                  column_transformer=column_transformer,
                                                                  preprocessing_pipeline=preprocessing_pipeline,
                                                                  count_per_step=count_per_step,
                                                                  baseline_accuracy=baseline_accuracy)

            accuracies.append(this_step_accuracies)
        if plot:
            plt.plot(np.linspace(0, 100, resolution), accuracies)
            plt.xlabel('Percentile of perturbation range', fontsize=13)
            plt.ylabel('Loss of accuracy', fontsize=13)
        return accuracies

    def gradual_perturbation_step(self, model, X, y, baseline_accuracy, importances, column_transformer,
                                  preprocessing_pipeline=None,
                                  count_per_step=5):
        transformers_for_update = [[t[0], t[2]] for t in column_transformer.transformers if
                                   '_INXAI_' in t[0] and hasattr(t[1], 'set_importances')]
        for t, c in transformers_for_update:
            column_transformer.set_params(**{t + '__importances': importances[[X.columns.get_loc(ci) for ci in c]]})

        this_step_accuracies = []
        for i in range(count_per_step):
            perturbed_dataset = column_transformer.fit_transform(X)
            colnames = [c.replace(t + "__", "") for c in column_transformer.get_feature_names()
                        for t, _ in transformers_for_update]

            if preprocessing_pipeline is None:
                dataset = pd.DataFrame(perturbed_dataset, columns=colnames)
            else:
                dataset = preprocessing_pipeline.fit_transform(pd.DataFrame(perturbed_dataset, columns=colnames))
            predictions = model.predict(dataset)
            this_step_accuracies.append(accuracy_score(y, predictions))
        return baseline_accuracy - np.mean(this_step_accuracies)

    def gradual_elimination(self):
        """
        Perturb one variable at a time according to importance and calculate given metric (acc only supported)
        :return:
        """
        pass

    def stability(self, X, all_importances, epsilon=3,perturber=None, perturber_strategy='mean', dissimilarity='euclidean', confidence=None):
        """Stability as Lipschitz coefficient.

        :param X:
        :param all_importances:
        :param epsilon:
        :return:
        """
        l_values = []

        if  not isinstance(all_importances, np.ndarray):
            all_importances = np.array(all_importances)

        if confidence is None:
            confidence = np.ones(all_importances.shape[0])

        for data_idx, (_, observation) in enumerate(X.iterrows()):
            max_val = 0
            for idx, (_, other_observation) in enumerate(X.iterrows()):
                dist = np.linalg.norm(observation - other_observation)
                if dist < epsilon:
                    l_val = np.linalg.norm(
                        pd.core.series.Series(all_importances[data_idx]) - pd.core.series.Series(
                            all_importances[idx])) / (dist+1) * confidence[data_idx]
                    if l_val > max_val:
                        max_val = l_val
            #if max_val:
            l_values.append(1/(max_val+1))
        return l_values

    def consistency(self, all_importances_per_model, perturber=None, perturber_strategy='mean', dissimilarity='euclidean', confidence =None):
        """ Calculates maximum distance to explanation generated by the same instance and different models

        :param all_importances_per_model:
        :return:
        """
        c_values = []

        if  not isinstance(all_importances_per_model, np.ndarray):
            all_importances_per_model = np.array(all_importances_per_model)

        if confidence is None:
            confidence = np.ones((all_importances_per_model.shape[0],all_importances_per_model.shape[1]))
        elif not isinstance(confidence,np.ndarray):
            confidence = np.array(confidence).reshape((all_importances_per_model.shape[0],all_importances_per_model.shape[1]))

        for obs_idx in range(len(all_importances_per_model[0])):
            largest_dist = 0
            for model_idx, model_imps in enumerate(all_importances_per_model):
                for other_idx, compared_model in enumerate(all_importances_per_model[:model_idx]):
                    current_imps = model_imps[obs_idx]
                    other_imps = compared_model[obs_idx]
                    if not isinstance(current_imps, np.ndarray):
                        current_imps = np.array(current_imps)
                    if not isinstance(other_imps, np.ndarray):
                        other_imps = np.array(other_imps)
                    dist = np.linalg.norm(current_imps - other_imps)*confidence[model_idx,obs_idx]*confidence[other_idx,obs_idx]
                    if dist > largest_dist:
                        largest_dist = dist
            c_values.append(1.0/(largest_dist+1))

        return c_values








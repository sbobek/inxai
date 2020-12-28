from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def create_intermediate_points(start_vals, end_vals, resolution):
    arr = []
    for start_val, end_val in zip(start_vals, end_vals):
        arr.append(np.linspace(start_val, end_val, resolution))
    return np.array(arr).T


class GlobalFeatureMetric:
    """

    """

    def gradual_perturbation(self, model, X, y, importances_orig, column_transformer, preprocessing_pipeline=None,
                             resolution=10, count_per_step=5, mode='all'):
        """


        :param model:
        :param X:
        :param y:
        :param importances_orig:
        :param column_transformer:
        :param preprocessing_pipeline:
        :param resolution:
        :param count_per_step:
        :param mode:
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
        pass

    def stability(self):
        pass

    def consistency(self):
        pass

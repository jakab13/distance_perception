import pandas as pd
from result_file_functions import get_solutions_and_responses, get_stage_results
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


solutions_responses = get_solutions_and_responses(root_path='C:/Users/mariu/OneDrive/Desktop/behavioural_analysis/data/result_files')


def create_cm(solutions_responses=solutions_responses, overall=False):
    if not overall:
        for id in solutions_responses:
            scores = pd.DataFrame()
            for tests in solutions_responses[id]:
                scores = scores._append(solutions_responses[id][tests])
            c_matrix = metrics.confusion_matrix(scores['solutions'], scores['responses'])
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                                                        display_labels=unique_labels(scores['solutions']))
            acc = round(100 * metrics.accuracy_score(scores['solutions'], scores['responses']), 2)
            mse = round(metrics.mean_squared_error(scores['solutions'], scores['responses']), 2)
            fig, ax = plt.subplots(nrows=1, ncols=1)
            cm_display.plot(ax=ax)
            fig.suptitle('Confusion Matrix all tests combined')
            ax.set_title('Accuracy: ' + str(acc) + '%', loc='left', fontsize='small')
            ax.set_title('Mean squared error: ' + str(mse), loc='right', fontsize='small')
            ax.set_title(id)
            ax.set_xlabel('Response')
            ax.set_ylabel('Solution')
    if overall:
        scores = pd.DataFrame()
        for id in solutions_responses:
            for tests in solutions_responses[id]:
                scores = scores._append(solutions_responses[id][tests])
        c_matrix = metrics.confusion_matrix(scores['solutions'], scores['responses'])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                                                    display_labels=unique_labels(scores['solutions']))
        acc = round(100 * metrics.accuracy_score(scores['solutions'], scores['responses']), 2)
        mse = round(metrics.mean_squared_error(scores['solutions'], scores['responses']), 2)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        cm_display.plot(ax=ax)
        fig.suptitle('Confusion Matrix all participants and tests combined')
        ax.set_title('Accuracy: ' + str(acc) + '%', loc='left', fontsize='small')
        ax.set_title('Mean squared error: ' + str(mse), loc='right', fontsize='small')
        ax.set_xlabel('Response')
        ax.set_ylabel('Solution')



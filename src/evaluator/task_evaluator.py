def evaluate_nli(llm_outputs: list, true_labels: list):
    evaluate_nli = []
    for output, label in zip(llm_outputs, true_labels):
        if output == label:
            evaluate_nli.append(1)
        elif output != label:
            evaluate_nli.append(0)


def evaluate_sa(llm_outputs: list, true_labels: list):
    evaluate_sa = []
    for output, label in zip(llm_outputs, true_labels):
        if output == label:
            evaluate_sa.append(1)
        elif output != label:
            evaluate_sa.append(0)

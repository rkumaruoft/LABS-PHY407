import numpy as np
from Lab02_Q1_b import method_1_std, method_2_std, get_relative_error

if __name__ == "__main__":
    mean_1, sigma_1, n_1 = 0., 1., 2000
    mean_2, sigma_2, n_2 = 1.e7, 1., 2000

    sequence_1 = np.random.normal(mean_1, sigma_1, n_1)
    sequence_2 = np.random.normal(mean_2, sigma_2, n_2)

    # for sequence 1
    std1_method_1 = method_1_std(sequence_1)
    std1_method_2 = method_2_std(sequence_1)
    std1_real = np.std(sequence_1, ddof=1)
    seq1_error_1 = get_relative_error(std1_method_1, std1_real)
    seq1_error_2 = get_relative_error(std1_method_2, std1_real)
    print(f"Sequence 1 std real: {std1_real}")
    print(f"Sequence 1 - Method 1 std:{std1_method_1} Error: {seq1_error_1}")
    print(f"Sequence 1 - Method 2 std:{std1_method_2} Error: {seq1_error_2}")

    # for sequence 2
    std2_method_1 = method_1_std(sequence_2)
    std2_method_2 = method_2_std(sequence_2)
    std2_real = np.std(sequence_2, ddof=1)
    seq2_error_1 = get_relative_error(std2_method_1, std2_real)
    seq2_error_2 = get_relative_error(std2_method_2, std2_real)
    print(f"Sequence 2 std real: {std2_real}")
    print(f"Sequence 2 - Method 1 std:{std2_method_1} Error: {seq2_error_1}")
    print(f"Sequence 2 - Method 2 std:{std2_method_2} Error: {seq2_error_2}")



import random
import numpy as np
import itertools
import statsmodels.api as sm
import scipy.stats as stats

def generate_shuffled_list(size, numbers):
  """Generates a shuffled list with an equal number of each number in 'numbers'.

  Args:
    size: The total size of the list (must be a multiple of the length of 'numbers').
    numbers: A list of numbers to include in the shuffled list.

  Returns:
    A list with an equal number of each number in 'numbers' in random order.
  """
  num_categories = len(numbers)

  if size % num_categories != 0:
    raise ValueError("Size must be a multiple of the number of categories to have an equal number of each.")

  category_size = size // num_categories
  my_list = []
  for num in numbers:
      my_list.extend([num] * category_size)
  random.shuffle(my_list)
  return my_list

def choose_different_array(array_of_arrays, correct_mapping):
  """Chooses a random array from array_of_arrays, excluding correct_mapping.

  Args:
    array_of_arrays: The array of arrays to choose from.
    correct_mapping: The array to exclude.

  Returns:
    A randomly chosen array from array_of_arrays that is not equal to correct_mapping.
  """
  # Filter out the correct_mapping array
  filtered_arrays = [arr for arr in array_of_arrays if not np.array_equal(arr, correct_mapping)]

  # Choose a random array from the filtered list
  if filtered_arrays:
    chosen_array = random.choice(filtered_arrays)
  else:
    # Handle the case where all arrays are equal to correct_mapping
    chosen_array = None  # Or raise an exception if appropriate

  return chosen_array

def action_softmax(action_func, beta):
    scale = np.sum([np.exp(beta * f) for f in action_func.values()])
    prob = np.zeros(len(action_func))
    for a, f in action_func.items():
        prob[a] = np.exp(beta * f) / scale

    return prob

# --- Generate two beta distributions with different means but similar variances and custom bounds ---
def generate_beta_with_diff_means_sim_vars(mean1, mean2, var, sz, a1=0, b1=1):
    """
    Generates two beta distributions with different means but similar variances,
    adjusted to fit within specified bounds [a1, b1] and [a2, b2].
    """

    # Compute alpha and beta for the original (0,1) Beta distribution
    def compute_alpha_beta(mean, var):
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        return alpha, beta

    # Adjusted mean for a Beta distribution within [a, b]
    def adjust_mean(mean, a, b):
        return (mean - a) / (b - a)

    # First distribution
    adj_mean1 = adjust_mean(mean1, a1, b1)
    alpha1, beta1 = compute_alpha_beta(adj_mean1, var)
    dist1 = stats.beta.rvs(alpha1, beta1, size=sz)
    dist1 = a1 + (b1 - a1) * dist1  # Transform to new range

    # Second distribution
    adj_mean2 = adjust_mean(mean2, a1, b1)
    alpha2, beta2 = compute_alpha_beta(adj_mean2, var)
    dist2 = stats.beta.rvs(alpha2, beta2, size=sz)
    dist2 =  a1 + (b1 - a1) * dist2  # Transform to new range

    # Compute 95% confidence interval bounds
    lower_bound1, upper_bound1 = np.percentile(dist1, [2.5, 97.5])
    lower_bound2, upper_bound2 = np.percentile(dist2, [2.5, 97.5])

    return dist1, dist2, alpha1, beta1, alpha2, beta2

def generate_valid_mappings(num_stimuli, num_actions):
    # Create a list of possible actions
    actions = list(range(num_actions))

    # Generate all possible assignments of actions to stimuli
    all_possible_mappings = itertools.product(actions, repeat=num_stimuli)
    lower, upper = (1, 3) if num_stimuli == 6 else (0, 1)
    valid_mappings = []
    for mapping in all_possible_mappings:
        # Count occurrences of each action in the mapping
        action_counts = [mapping.count(action) for action in actions]

        # Check if each action is used at least once and at most three times
        if all(lower <= count <= upper for count in action_counts):
            valid_mappings.append(mapping)

    return valid_mappings    
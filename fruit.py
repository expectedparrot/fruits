from edsl import QuestionList, ScenarioList, QuestionFreeText
import random 

random.seed("deepmind")
q = QuestionList(question_text = "Generate a list of 10 fruits", question_name = "fruits")
fruits = q()

def list_to_pref(items):
    "Format a list of items as a preference ordering e.g., A > B > C from ['A', 'B', 'C']"
    return  " > ".join(items)

true_preferences = [random.sample(fruits, len(fruits)) for _ in range(20)]
preference_ordering = [list_to_pref(p) for p in true_preferences]

# Create a scenario list with the preference ordering and the true preferences
scenarios = (ScenarioList
             .from_list("preference_ordering", preference_ordering)
             .add_list("true_preference", true_preferences)
)

# Ask the agent to generate a paragraph describing the preferences
q = QuestionFreeText(
    question_text = """Imagine a person is describing their taste in fruits to someone who will shop for them. 
    Generate one sentence of text that try to capture this person's preferences for fruits, without mentioning the fruits by name. 
    It is OK to mention dislikes and negative opinions.
    Their preferences are: {{ preference_ordering }}
    """,
    question_name = "pref_paragraph")

results = q.by(scenarios).run()

# print them
results.select("pref_paragraph", 'preference_ordering').print(format = "rich")

# Use the inferred preferences to generate a new scenario where another agent take the paragraph as input 
# and tries to learn the preference ordering
new_scenario = (results
                .select("pref_paragraph", 'true_preference')
                .to_scenario_list()
                .add_list("fruits", [fruits for _ in range(20)])
                )

q_infer = QuestionList(question_text = """A person described their preferences over fruits as {{ pref_paragraph }}.
                The collection of fruits is: '{{ fruits }}'.
                Based on what they said, what do you guess is their rank-ordering of the fruits? 
                """, 
                question_name = "inferred_preference")

results_inference = q_infer.by(new_scenario).run()

# Compare to what the agent did to what we would expect with random guessing
def levenshtein_distance(list1, list2):
    """Compute the Levenshtein distance (edit distance) between two lists."""
    n, m = len(list1), len(list2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[n][m]

distances = [levenshtein_distance(inferred, actual) 
             for inferred, actual in results_inference.select("inferred_preference", "true_preference").to_list()]

average = sum(distances)/len(distances)
print(f"Average Levenshtein distance: {average:.2f}")

def random_distance():
    return levenshtein_distance(random.sample(fruits, len(fruits)), random.sample(fruits, len(fruits)))

N = 1000
average_random = sum([random_distance() for _ in range(N)])/N
print(f"Average random distance: {average_random:.2f}")
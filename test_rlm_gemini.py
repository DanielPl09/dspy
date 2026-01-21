"""
Minimal test script for dspy.RLM with Google Gemini API
This script keeps API calls minimal by using:
- Small test dataset instead of 16-month slack dump
- Low max_iterations and max_llm_calls limits
"""

import dspy

# Configure dspy to use Google Gemini with the provided API key
# Note: Use "gemini/" prefix for Google Gemini models
google_lm = dspy.LM("gemini/gemini-1.5-flash", api_key="AIzaSyAqfZRxJbxPfElPxVi79L9r0zBDCyEaZTc")
dspy.configure(lm=google_lm)

# Minimal mock slack discussion (instead of 16 months of real data)
# This keeps token usage and API calls minimal for testing
slack_dump = """
2024-01-15 Omar: What if we built a tool that automatically optimizes prompts using feedback?
2024-01-15 Isaac: Love it! We could use the model itself to suggest improvements.

2024-02-03 Isaac: Been thinking about a DSPy compiler that uses reinforcement learning.
2024-02-03 Omar: That's interesting. Maybe combine it with the prompt optimization idea?

2024-03-12 Omar: We should explore multi-hop reasoning with recursive LMs.
2024-03-12 Isaac: Yes! And maybe add caching for intermediate results.

2024-04-20 Isaac: What about a visual debugger for DSPy programs?
2024-04-20 Omar: That would be super useful. Could show the trace of module calls.

2024-05-08 Omar: Idea: automatic dataset generation from unlabeled data.
2024-05-08 Isaac: Ooh, using the LM to bootstrap examples. Let's revisit this.
"""

# Create RLM with signature matching the original code
# Keep max_iterations and max_llm_calls LOW to minimize API usage
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=5,      # Reduced from default 20
    max_llm_calls=10,      # Reduced from default 50
    verbose=True           # Show what's happening
)

print("=" * 80)
print("Testing dspy.RLM with Google Gemini")
print("=" * 80)
print()

# Run the RLM
output = rlm(
    discussion=slack_dump,
    request="What are the 5 coolest unfinished ideas that Omar & Isaac keep coming back to?"
)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print("Ideas found:")
for i, idea in enumerate(output.ideas, 1):
    print(f"{i}. {idea}")

print()
print("First idea:", output.ideas[0])

# Optional: Print the trajectory to see how RLM reasoned through the problem
print()
print("=" * 80)
print("EXECUTION TRAJECTORY (for debugging)")
print("=" * 80)
print(output.trajectory)

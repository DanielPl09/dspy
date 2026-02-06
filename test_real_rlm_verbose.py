#!/usr/bin/env python3
"""
ACTUAL dspy.RLM test with verbose output to see real sub-LLM calls.

We'll use verbose=True to see the exact code RLM writes and try to get
the interpreter working to see actual llm_query() calls.
"""

import dspy
import os
import sys

# Try to help with Deno/interpreter issues
os.environ.setdefault('DENO_DIR', '/tmp/deno_cache')

def test_real_rlm_with_verbose():
    """Test real RLM with verbose to see code and sub-LLM calls."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)

    print("="*80)
    print("REAL dspy.RLM Test - Showing Actual Code and Sub-LLM Calls")
    print("="*80)

    # Create documents that are LONG enough to require llm_query
    # Make each doc substantial so RLM can't just regex parse
    documents = {
        "The Social Network Film": """
        The Social Network is a 2010 American biographical drama film directed by David Fincher.
        The screenplay was adapted from Ben Mezrich's book The Accidental Billionaires by Aaron Sorkin.
        The film depicts the founding of social networking website Facebook and the resulting lawsuits.
        It stars Jesse Eisenberg as founder Mark Zuckerberg, along with Andrew Garfield and Justin Timberlake.
        The film was produced by Scott Rudin, Dana Brunetti, Michael De Luca, and Ceán Chaffin.
        It received widespread acclaim, with critics praising the editing, acting, score, and Fincher's direction.
        The film won three Academy Awards for Best Adapted Screenplay, Best Original Score, and Best Film Editing.
        It was also nominated for Best Picture and Best Director at the 83rd Academy Awards.
        The film grossed $224.9 million worldwide on a budget of $40 million.
        It is considered one of the best films of the 2010s.
        """,

        "David Fincher Biography": """
        David Andrew Leo Fincher (born August 28, 1962) is an American film director.
        Fincher was born in Denver, Colorado, and raised in Marin County, California.
        He began his career in the 1980s, working at Industrial Light & Magic and directing commercials.
        His feature film directorial debut was Alien 3 (1992), which was not well received.
        He found success with Se7en (1995), a dark thriller starring Brad Pitt and Morgan Freeman.
        His other notable films include The Game (1997), Fight Club (1999), Panic Room (2002),
        Zodiac (2007), The Curious Case of Benjamin Button (2008), The Social Network (2010),
        The Girl with the Dragon Tattoo (2011), Gone Girl (2014), and Mank (2020).
        He has been nominated for the Academy Award for Best Director multiple times.
        Fincher is known for his meticulous attention to detail and dark, psychological themes.
        """,

        "Aaron Sorkin Information": """
        Aaron Benjamin Sorkin (born June 9, 1961) is an American playwright, screenwriter, and film director.
        He was born in Manhattan, New York City, to a Jewish family.
        Sorkin attended Scarsdale High School and later Syracuse University.
        He is known for his fast-paced, witty dialogue and emphasis on idealism.
        His breakthrough came with the play A Few Good Men in 1989.
        He created the television series The West Wing (1999-2006), which won multiple Emmy Awards.
        Other TV work includes Sports Night and The Newsroom.
        He wrote the screenplay for The Social Network (2010), for which he won an Academy Award.
        Other notable screenplays include Moneyball (2011), Steve Jobs (2015), and Molly's Game (2017).
        Sorkin made his directorial debut with Molly's Game.
        """,

        "Facebook Company": """
        Facebook is an American online social media and social networking service owned by Meta Platforms.
        It was founded in 2004 by Mark Zuckerberg while he was a student at Harvard University.
        The company went public in May 2012 with an initial public offering that raised $16 billion.
        Facebook had 2.91 billion monthly active users as of Q1 2022.
        The platform allows users to post content, share photos and videos, and interact with friends.
        In 2012, Facebook acquired Instagram for $1 billion.
        In 2014, it acquired WhatsApp for $19 billion and Oculus VR for $2 billion.
        The company has faced controversies regarding privacy, misinformation, and data breaches.
        In 2021, Facebook rebranded its parent company to Meta Platforms to focus on the metaverse.
        Mark Zuckerberg remains the CEO of Meta Platforms.
        """
    }

    # Combine into context with clear instruction to use llm_query
    context = """IMPORTANT: You have 4 separate documents below. Each document is LONG and contains different information.
You MUST use llm_query() to analyze each document separately, as they are too long to parse manually.
DO NOT try to use regex or string parsing - use llm_query() for semantic analysis.

"""
    for title, content in documents.items():
        context += f"\n=== DOCUMENT: {title} ===\n{content}\n"

    question = "What year was the director of 'The Social Network' born?"

    print(f"\nQuestion: {question}")
    print(f"\nContext has {len(documents)} documents")
    print("\nRunning RLM with verbose=True to see the code it writes...")
    print("="*80)

    # Create RLM with verbose to see code
    rlm = dspy.RLM(
        signature="context, question -> answer: str",
        max_iterations=20,
        max_llm_calls=15,  # Encourage use of llm_query
        verbose=True,  # SEE THE CODE!
    )

    # Run RLM
    result = rlm(context=context, question=question)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Answer: {result.answer}")
    print(f"Iterations used: {len(result.trajectory)}")

    # Analyze the trajectory to count llm_query usage
    llm_query_count = 0
    llm_query_batched_count = 0

    print("\n" + "="*80)
    print("ANALYZING CODE FOR llm_query CALLS")
    print("="*80)

    for i, step in enumerate(result.trajectory, 1):
        code = step.get('code', '')

        # Count different types of calls
        single_calls = code.count('llm_query(')
        batch_calls = code.count('llm_query_batched(')

        if single_calls > 0 or batch_calls > 0:
            print(f"\nIteration {i}:")
            print(f"  llm_query() calls: {single_calls}")
            print(f"  llm_query_batched() calls: {batch_calls}")
            print(f"  Code snippet:")
            # Show relevant lines
            for line in code.split('\n'):
                if 'llm_query' in line:
                    print(f"    {line.strip()}")

        llm_query_count += single_calls
        llm_query_batched_count += batch_calls

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total llm_query() calls in code: {llm_query_count}")
    print(f"Total llm_query_batched() calls in code: {llm_query_batched_count}")
    print(f"Total iterations: {len(result.trajectory)}")

    if llm_query_count == 0 and llm_query_batched_count == 0:
        print("\n⚠️  WARNING: RLM did not use llm_query()!")
        print("This might be because:")
        print("  1. The interpreter couldn't execute the code")
        print("  2. RLM found a simpler approach (regex/parsing)")
        print("  3. The documents were too short")
        print("\nCheck the verbose output above to see what code RLM wrote.")
    else:
        print("\n✓ RLM used llm_query() for sub-LLM calls!")

    return result


if __name__ == "__main__":
    test_real_rlm_with_verbose()

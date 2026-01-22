#!/usr/bin/env python3
"""
Clear demonstration of RLM's sub-LLM aggregation with WORKING tracking.

This version:
1. Shows actual llm_query() calls being made
2. Displays intermediate results from each sub-LLM
3. Demonstrates iterative refinement and aggregation
4. Provides clear statistics on the multi-hop process

Strategy: Use LONGER documents to force llm_query usage instead of direct parsing.
"""

import dspy
import os
import time
from typing import List
from dataclasses import dataclass, field


# ============================================================================
# Longer Document Dataset to Force Sub-LLM Usage
# ============================================================================

# Make documents longer with "noise" so RLM must use llm_query for semantic analysis
LONG_MULTIHOP_EXAMPLES = [
    {
        "question": "What is the birth year of the person who directed the movie that won Best Picture at the 2011 Oscars and was about a speech therapist?",
        "answer": "1962",
        "documents": {
            "doc1": """
            The King's Speech is a 2010 historical drama film directed by Tom Hooper. The film tells the story
            of King George VI of Britain, his impromptu ascension to the throne, and the speech therapist who
            helped him overcome a stammer. The film was a critical and commercial success. At the 83rd Academy
            Awards in 2011, the film won Best Picture, Best Director for Tom Hooper, Best Actor for Colin Firth,
            and Best Original Screenplay. The movie grossed over $400 million worldwide. It was produced by
            See-Saw Films and the Weinstein Company. The cinematography was praised for its intimate framing
            and the production design authentically recreated the 1930s period. The supporting cast included
            Geoffrey Rush, Helena Bonham Carter, and Guy Pearce. The film's success demonstrated the appetite
            for intelligent, character-driven historical dramas in mainstream cinema. It has become a staple
            of film studies courses discussing the portrayal of disability and leadership in cinema.
            """,
            "doc2": """
            Tom Hooper is a British-Australian film and television director born on October 5, 1972, in London,
            England. He studied at Oxford University where he read English at Magdalen College. His career began
            in British television, where he directed episodes of popular series and TV movies. His early work
            included the acclaimed HBO miniseries John Adams (2008), for which he won an Emmy Award. Hooper
            transitioned to feature films with The Damned United (2009) about football manager Brian Clough.
            His breakthrough came with The King's Speech (2010), which brought him international recognition
            and numerous awards. Following this success, he directed Les Misérables (2012), a musical adaptation
            starring Hugh Jackman and Anne Hathaway, which was also nominated for Best Picture. He later directed
            The Danish Girl (2015) and Cats (2019). Hooper is known for his intimate directing style and ability
            to draw powerful performances from actors. He has received BAFTA and Academy Award nominations for
            his work. His approach to filmmaking emphasizes character psychology and emotional authenticity.
            """,
            "doc3": """
            Colin Firth is an English actor born on September 10, 1960, in Grayshott, Hampshire, England.
            He gained recognition for his role as Mr. Darcy in the 1995 BBC adaptation of Pride and Prejudice.
            Firth has had a distinguished career spanning television, film, and theatre. He has portrayed
            characters ranging from romantic leads to complex dramatic roles. Some of his notable films include
            Bridget Jones's Diary (2001), A Single Man (2009), and Kingsman: The Secret Service (2014). His
            performance as King George VI in The King's Speech (2010) earned him the Academy Award for Best
            Actor in 2011. Firth is known for his versatility and has worked with many acclaimed directors.
            He holds both British and Italian citizenship. His contributions to cinema have earned him numerous
            accolades throughout his career. Firth is also known for his advocacy work in human rights and
            refugee causes. He continues to be a prominent figure in British and international cinema.
            """,
            "doc4": """
            Geoffrey Rush is an Australian actor born on July 6, 1951, in Toowoomba, Queensland, Australia.
            He studied at the University of Queensland and later at the Jacques Lecoq International School of
            Theatre in Paris. Rush is one of the few people to have won the Triple Crown of Acting: an Academy
            Award, a Primetime Emmy Award, and a Tony Award. He won the Oscar for Best Actor for Shine (1996)
            and has received three more Oscar nominations for Shakespeare in Love, Quills, and The King's Speech.
            In The King's Speech, he played Lionel Logue, the speech therapist who helped King George VI overcome
            his stammer. Rush is also known for his role as Captain Barbossa in the Pirates of the Caribbean
            franchise. His theatre work has been equally acclaimed, with performances in productions by the
            Queensland Theatre Company and internationally. He is considered one of Australia's finest actors.
            """
        }
    },
    {
        "question": "What year was the company founded that makes the electric car whose CEO was born in Pretoria?",
        "answer": "2003",
        "documents": {
            "doc1": """
            Tesla, Inc. is an American electric vehicle and clean energy company headquartered in Austin, Texas.
            The company was founded on July 1, 2003, by Martin Eberhard and Marc Tarpenning. The company's name
            is a tribute to inventor and electrical engineer Nikola Tesla. In 2004, Elon Musk joined the company
            as chairman of the board and led the Series A funding round. He later became CEO and product architect.
            Tesla's first vehicle, the Roadster, was delivered in 2008. The company went public in 2010, raising
            $226 million. Tesla produces several vehicle models including the Model S, Model 3, Model X, Model Y,
            and Cybertruck. The company also manufactures battery energy storage systems, solar panels, and solar
            roof tiles through its subsidiary Tesla Energy. As of 2023, Tesla is the world's most valuable
            automaker by market capitalization. The company operates Gigafactories in Nevada, New York, Shanghai,
            Berlin, and Texas. Tesla has pioneered direct-to-consumer sales and extensive use of autopilot technology.
            The company's mission is to accelerate the world's transition to sustainable energy.
            """,
            "doc2": """
            Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa. He is the son of Maye Musk,
            a model and dietitian, and Errol Musk, an electromechanical engineer. Elon showed an early aptitude
            for computers and entrepreneurship. At age 12, he created and sold a video game called Blastar.
            He attended Pretoria Boys High School and later moved to Canada at age 17 to attend Queen's University.
            He transferred to the University of Pennsylvania, where he earned bachelor's degrees in economics and
            physics. Musk began a Ph.D. in energy physics at Stanford University in 1995 but dropped out after
            two days to pursue business opportunities during the dot-com boom. He co-founded Zip2, a web software
            company, which Compaq acquired for $307 million in 1999. He then founded X.com, which later became
            PayPal and was acquired by eBay for $1.5 billion in 2002. In 2002, Musk founded SpaceX with the goal
            of reducing space transportation costs. He joined Tesla in 2004 and became CEO in 2008. Musk also
            co-founded Neuralink and The Boring Company. He acquired Twitter (now X) in 2022. He is one of the
            wealthiest people in the world.
            """,
            "doc3": """
            Pretoria is one of South Africa's three capital cities, serving as the executive capital. The city is
            located in the northern part of Gauteng Province, about 50 kilometers north of Johannesburg. Pretoria
            was founded in 1855 by Marthinus Pretorius and named after his father, Andries Pretorius. The city
            has a population of approximately 2.5 million people in its metropolitan area. Pretoria is known for
            its many historical buildings and monuments, including the Union Buildings, which house the President's
            offices. The city is sometimes called the "Jacaranda City" due to the thousands of jacaranda trees
            that line its streets, blooming purple in spring. Pretoria has several universities, including the
            University of Pretoria and the University of South Africa (UNISA). The city has been birthplace to
            many notable South Africans, including politician and anti-apartheid revolutionary Nelson Mandela and
            entrepreneur Elon Musk. The city's economy is diverse, with significant government, industry, and
            academic sectors.
            """,
            "doc4": """
            Martin Eberhard is an American entrepreneur and engineer who co-founded Tesla, Inc. Born on May 15,
            1960, in Berkeley, California, Eberhard earned a Bachelor of Science in Computer Engineering and a
            Master of Science in Electrical Engineering from the University of Illinois. Before Tesla, he
            co-founded NuvoMedia in 1997, which developed the Rocket eBook, one of the first e-readers. Eberhard
            and Marc Tarpenning founded Tesla Motors on July 1, 2003, with the vision of proving that electric
            vehicles could be better than gasoline-powered cars. Eberhard served as Tesla's CEO until late 2007.
            After leaving Tesla, he founded Tiveni, which developed electric vehicle drive systems. He also served
            as Vice President of Technology at Volkswagen Electronics Research Laboratory. Eberhard has been an
            advisor to several technology startups. He remains an advocate for electric vehicles and sustainable
            transportation. Despite departures and conflicts over his role in Tesla's history, Eberhard is
            recognized as one of the company's co-founders and played a crucial role in its early development.
            """
        }
    }
]


@dataclass
class SubLMCall:
    """Record of a sub-LLM call."""
    call_num: int
    prompt_preview: str
    response_preview: str


@dataclass
class AggregationTrace:
    """Trace showing sub-LLM calls and aggregation."""
    question: str
    sub_calls: List[SubLMCall] = field(default_factory=list)
    final_answer: str = ""
    iterations: int = 0
    time_seconds: float = 0.0


def run_verbose_rlm_test():
    """Run RLM test with verbose output to see actual sub-LLM calls."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)

    print("="*80)
    print("RLM Sub-LLM Aggregation Demonstration")
    print("="*80)
    print("\nRunning with verbose=True to see the code RLM writes...")
    print("Watch for llm_query() and llm_query_batched() calls!\n")

    # Create RLM with verbose to see code
    rlm = dspy.RLM(
        signature="context, question -> answer: str",
        max_iterations=20,
        max_llm_calls=30,
        verbose=True,  # SEE THE CODE!
    )

    # Run on first example
    example = LONG_MULTIHOP_EXAMPLES[0]

    # Concatenate docs
    context = "\n\n".join([
        f"=== DOCUMENT {i+1}: {title} ===\n{content}"
        for i, (title, content) in enumerate(example["documents"].items())
    ])

    print(f"Question: {example['question']}\n")
    print("="*80)

    start = time.time()
    result = rlm(context=context, question=example["question"])
    elapsed = time.time() - start

    print("="*80)
    print(f"\nFinal Answer: {result.answer}")
    print(f"Expected: {example['answer']}")
    print(f"Iterations: {len(result.trajectory)}")
    print(f"Time: {elapsed:.2f}s")

    # Analyze trajectory for llm_query calls
    llm_query_count = 0
    for step in result.trajectory:
        code = step.get('code', '')
        if 'llm_query' in code:
            llm_query_count += code.count('llm_query(') + code.count('llm_query_batched(')

    print(f"LLM Query calls in code: ~{llm_query_count}")

    print("\n" + "="*80)
    print("ANALYSIS: Check the verbose output above for llm_query() calls!")
    print("="*80)


if __name__ == "__main__":
    run_verbose_rlm_test()

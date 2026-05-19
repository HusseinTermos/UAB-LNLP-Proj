# def parse_response(raw_text: str, claim: str) -> dict:
#     """
#     Attempt to extract valid JSON from the model output.
#     Falls back to a minimal structure using the raw claim
#     if parsing fails.
#     """
#     # direct parse
#     try:
#         return json.loads(raw_text.strip())
#     except json.JSONDecodeError:
#         pass

#     # extract JSON block if model added surrounding text
#     match = re.search(r"\{.*\}", raw_text, re.DOTALL)
#     if match:
#         try:
#             return json.loads(match.group())
#         except json.JSONDecodeError:
#             pass

#     # fallback
#     logger.warning(f"JSON parse failed. Using fallback for: {claim[:60]}")
#     return {
#         "normalized_claim": claim,
#         "pico": {
#             "population":   "general population",
#             "intervention": claim,
#             "outcome":      "",
#         },
#         "queries":       [claim],
#         "bm25_keywords": " ".join(claim.split()),
#     }


# def validate(result: dict) -> bool:
#     required = ["normalized_claim", "pico", "queries", "bm25_keywords"]
#     if not all(k in result for k in required):
#         return False
#     if not isinstance(result["queries"], list) or len(result["queries"]) == 0:
#         return False
#     pico_keys = ["population", "intervention", "outcome"]
#     if not all(k in result.get("pico", {}) for k in pico_keys):
#         return False
#     return True

# In the future, we can encorporate more context when reformulating such as the things below:

# if not validate(result):
#     logger.warning(f"Validation failed, using fallback for: {claim[:60]}")
#     result = {
#         "normalized_claim": claim,
#         "pico": {
#             "population":   "general population",
#             "intervention": claim,
#             "outcome":      "",
#         },
#         "queries":       [claim],
#         "bm25_keywords": " ".join(claim.split()),
#     }


        # print(f"Normalized:\n  {result.get('normalized_claim', 'N/A')}")
        # pico = result.get("pico", {})
        # print(f"PICO:")
        # print(f"  Population:   {pico.get('population',   'N/A')}")
        # print(f"  Intervention: {pico.get('intervention', 'N/A')}")
        # print(f"  Outcome:      {pico.get('outcome',      'N/A')}")
        # print(f"Queries:")
        # for i, q in enumerate(result.get("queries", []), 1):
        #     print(f"  {i}. {q}")
        # print(f"BM25 Keywords:\n  {result.get('bm25_keywords', 'N/A')}")
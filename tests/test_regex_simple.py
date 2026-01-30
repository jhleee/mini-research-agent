#!/usr/bin/env python3
"""Test heuristic regex in isolation."""
import re

query = '부산시와 순천시와 서귀포시'

# Match city names (with 시/도/군/구) OR standalone words
entities = re.findall(r'([가-힣]+(?:시|도|군|구)|(?:시|도|군|구)\b)', query)

print(f'Query: {query}')
print(f'Entities: {entities}')
print(f'Count: {len(entities)}')

if len(entities) >= 2:
    topics = [{'topic': e, 'description': f'{e} 인구 정보'} for e in entities]
    print(f'Separate topics: {topics}')
else:
    print('Only 1 entity found, returning fallback')

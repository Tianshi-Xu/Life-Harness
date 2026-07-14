import json

products = []
asins = set()
with open('/tmp/documents_100k.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        products.append(obj['product'])
        asins.add(obj['product'].get('asin', ''))

with open('/usr/src/webshop/data/items_shuffle_100k.json', 'w') as f:
    json.dump(products, f)

with open('/usr/src/webshop/data/items_ins_v2.json') as f:
    ins = json.load(f)
filtered = {k: v for k, v in ins.items() if k in asins}
with open('/usr/src/webshop/data/items_ins_v2_100k.json', 'w') as f:
    json.dump(filtered, f)

print(f'Generated {len(products)} products, {len(filtered)} instructions')

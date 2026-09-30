---
description: Python idiom rules (defaults, identity, iteration, dict lookups) with bad and good examples.
paths:
  - "**/*.py"
---
<!-- master: ai-code-review/rules/python-idioms.md; do not edit generated copies @ 68f1c972ce4a -->

## Never use a mutable default argument
Python evaluates the default once, when `def` runs, so every call that omits the argument mutates one shared object.
```python
# bad
def build_headers(token, headers={}):
    headers["Authorization"] = f"Bearer {token}"
    return headers  # one dict for every call: the next token overwrites the last
```
```python
# good
def build_headers(token, headers=None):
    merged = {} if headers is None else dict(headers)
    merged["Authorization"] = f"Bearer {token}"
    return merged
```
<!-- src: course -->

## Use `is` only for None and other singletons
`is` tests identity; two equal strings or numbers built at run time are often different objects.
```python
# bad
if order.status is "shipped":
    notify_customer(order)
```
```python
# good
if order.status == "shipped":
    notify_customer(order)
if order.tracking_code is None:
    request_tracking(order)
```
<!-- src: course -->

## Iterate over the object, not its indices
Index loops invite off-by-one errors and hide what the loop walks over.
```python
# bad
for i in range(len(sensors)):
    log_reading(sensors[i].label, sensors[i].value)
```
```python
# good
for sensor in sensors:
    log_reading(sensor.label, sensor.value)
for slot, sensor in enumerate(sensors, start=1):  # when the position matters
    log_reading(f"{slot}:{sensor.label}", sensor.value)
```
<!-- src: course -->

## Use `dict.get` for a missing key
`get` looks the key up once and names the fallback inline; note that it returns a stored None, not the fallback.
```python
# bad
locale = profile["locale"] if "locale" in profile else DEFAULT_LOCALE
```
```python
# good
locale = profile.get("locale", DEFAULT_LOCALE)
if plan_code not in PLANS:  # fine: branching on absence is not a default
    raise UnknownPlan(plan_code)
```
<!-- src: course -->

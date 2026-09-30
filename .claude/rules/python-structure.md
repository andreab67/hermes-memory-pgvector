---
description: Python structure rules (files, entry point, function scope, constants) with bad and good examples.
paths:
  - "**/*.py"
---
<!-- master: ai-code-review/rules/python-structure.md; do not edit generated copies @ 0fb4f9d14433 -->

## Open files with `with`
`with` releases the handle on every exit path, including an exception halfway through the block.
```python
# bad
handle = open(report_path, "w", encoding="utf-8")
handle.write(render(report))
handle.close()  # skipped if render() raises
```
```python
# good
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write(render(report))
```
<!-- src: course -->

## Guard the entry point with `__name__`
Module-level calls run on import, so a test or another module starts the app just by importing it.
```python
# bad
def sync_inventory():
    ...
sync_inventory()
```
```python
# good
if __name__ == "__main__":
    sync_inventory()
```
<!-- src: course, catalog D20 -->

## Let each function do one thing
Keep calculation apart from formatting, storage and I/O so each part has one reason to change and a test of its own.
```python
# bad
def checkout(cart):
    total = sum(item.price * item.qty for item in cart)
    print(f"Total due: {total}")
    ledger.append(total)
```
```python
# good
def cart_total(cart):
    return sum(item.price * item.qty for item in cart)
def total_line(total):
    return f"Total due: {total}"
```
<!-- src: course, catalog D19 -->

## Name constants instead of inlining numbers
Name each threshold, limit and rounding precision once so its meaning is stated and every use changes together (test data may stay literal).
```python
# bad
if temperature_c > 85.0:
    trip_alarm()
shown = round(voltage, 1)
```
```python
# good
ALARM_TEMP_C = 85.0
VOLTAGE_DECIMALS = 1
if temperature_c > ALARM_TEMP_C:
    trip_alarm()
shown = round(voltage, VOLTAGE_DECIMALS)
```
<!-- src: course, catalog D21 -->

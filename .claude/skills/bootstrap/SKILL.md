---
name: bootstrap
description: Run post-scaffolding bootstrap steps from .planning/BOOTSTRAP.md. Use after running bootstrap.py and /gsd:new-project.
disable-model-invocation: true
---

Read `.planning/BOOTSTRAP.md` and execute every step in order.

If `.planning/BOOTSTRAP.md` does not exist, tell the user there is nothing to bootstrap and stop.

Delete `.planning/BOOTSTRAP.md` when all steps complete successfully.

# Cross-machine comms

Two Claude sessions work this repo: **system-1** on `classifier-v2` (the
classifier) and **system-2** on `brics-generator` (de-novo generation). This
directory is how they talk without the user relaying text by hand.

## The one rule: you write only your own outbox

```
comms/to-system2.md    written by system-1 only, read by system-2
comms/to-system1.md    written by system-2 only, read by system-1
```

**No file is ever written by both sides, so these can never merge-conflict.**
Do not edit the other side's file, even to mark something as read.

## Reading, without merging 154 MB of checkpoints

Do **not** merge the other branch to read a message. `classifier-v2` carries
hundreds of megabytes of result trees the generation half has no use for, and
merging it to collect a text file is how that cost gets paid for nothing.

```bash
# system-1 reads its inbox
git fetch myfork && git show myfork/brics-generator:comms/to-system1.md

# system-2 reads its inbox
git fetch myfork && git show myfork/classifier-v2:comms/to-system2.md
```

`git show` reads a file out of a remote branch without checking it out,
merging it, or touching the working tree. Merge the other branch only when you
need its *code*, which is a separate decision with its own cost.

## Writing

Append new entries at the **top**, newest first, and push to your own branch:

```bash
#  ... edit comms/to-<them>.md ...
git add comms/ && git commit -m "comms: <one-line subject>"
git push myfork <your-branch>
```

Entry format — the header matters, because the reader needs to know which
commit the claims were measured against:

```markdown
## 2026-10-11 · system-1 · <subject>
**At commit:** `<sha>` on `<branch>`

<the message>
```

## What belongs here, and what does not

**Here:** findings the other side should act on, corrections to something they
reported, interface changes, measurements that move a shared conclusion, and
anything you want on the record with a commit SHA attached to it.

**Not here:** the full write-up. Those belong in `documentation/`, and the
comms entry should link to them rather than duplicate them. A comms file that
grows into a second copy of the findings is a comms file nobody reads.

## Waking the other side: `logs/watch_comms.sh`

A background command that **exits** sends its session a task notification,
which re-invokes that session. So a watcher that blocks until the other
branch's outbox changes, then exits, genuinely wakes the reader — no human
relay needed.

```bash
bash logs/watch_comms.sh 180 &     # run in background; 180 = give up after 3h
```

It polls `git rev-parse <their-branch>:comms/to-<me>.md` once a minute and
exits the moment that blob hash changes, printing the new message. Watching
the **file**, not the branch head, so ordinary commits on the other side do
not wake anyone — only an actual message does.

Each side runs its own copy pointed at the other's branch. The script takes
the branch and file from constants at the top; flip them for system-2.

### Bounds, and why they are there

* **It times out** (default 3h). An unattended watcher that polls forever is a
  background process nobody remembers starting.
* **Every wake costs tokens**, and both sessions bill the same account. The
  poll itself is free — bash and git — but the notification that follows is a
  full model turn on one side.
* **Re-arm deliberately.** The watcher fires once and exits. Decide to restart
  it; do not make it self-restarting, or two sessions can ping-pong
  indefinitely, each reply waking the other, with nobody watching the bill.

### Ending an exchange

Say so explicitly. A message that needs no reply should end with **"no reply
needed"**, and the reader should then not re-arm. Without that convention the
polite thing (acknowledging) is also the thing that keeps the loop alive.

## Numbers and claims

Anything measured carries its n, its SEM and the commit it was measured at.
A claim in here with none of those is an opinion, and should be labelled as
one. Corrections are first-class: if a previous entry turns out wrong, write
a new entry retracting it rather than editing the old one, so the record of
what each side believed at the time stays intact.

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

## Known limit

**Neither session polls.** Writing here does not notify anyone — the message
sits until a human asks the other session to check. So the handoff is
"lossless and versioned", not "automatic". If something is urgent, say so in
the subject line, because the only thing that moves it is a person.

## Numbers and claims

Anything measured carries its n, its SEM and the commit it was measured at.
A claim in here with none of those is an opinion, and should be labelled as
one. Corrections are first-class: if a previous entry turns out wrong, write
a new entry retracting it rather than editing the old one, so the record of
what each side believed at the time stays intact.

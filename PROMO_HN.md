# Hacker News - Show HN Post

## Title

Show HN: Claude Oracle – CLI to use Gemini as lead architect for Claude Code

## URL

https://github.com/n1ira/claude-oracle

---

## First Comment (post immediately after submitting)

Hey HN, I built this because I kept running into the same problem with Claude Code: it's excellent at writing code but sometimes makes questionable architectural decisions.

**The core idea:** Use Gemini as a "senior architect" that Claude consults for strategic decisions. Different training data, different blind spots.

**How it works:**

```bash
oracle ask "Should I use Redis or Postgres for job queue?"
oracle ask --files src/auth.py "Security review?"
```

The interesting part is `/fullauto` mode, a Claude Code slash command that enables high-autonomy operation:

```
/fullauto implement rate limiting for the API
```

Claude will:
1. Explore your codebase
2. Ask Gemini for an implementation plan
3. Execute it, checking back at decision points
4. Get final validation

You can give it a task and walk away.

**Technical details:**

- Maintains 5-exchange conversation history per project
- File attachments support line ranges (`file.py:10-50`)
- Image analysis for screenshots and diagrams
- Auto-provisions US Vast.ai instances for geo-restricted image generation (~$0.01/image)

**The meta part:** This repo was created using itself. I ran `/fullauto make this into a public GitHub repo` and it orchestrated the whole process.

Single-file Python, ~1700 lines, MIT license. Happy to answer questions about the implementation.

---

## Notes for posting:

- Post Tue-Thu around 9am EST for best visibility
- Keep the title under 80 chars
- First comment must go up within 60 seconds of posting
- Be ready to respond to comments for 12+ hours
- Don't be defensive about criticism
- If someone points out a bug, acknowledge and fix it publicly
- Avoid marketing speak - HN users smell BS immediately

# Twitter/X Thread Draft

## Tweet 1 (Hook)

Built a CLI that lets Claude Code consult Google's Gemini for architecture decisions.

It's like giving your AI coding assistant a senior architect to check in with.

Thread on how it works:

## Tweet 2

The problem: Claude is great at writing code, but sometimes goes off in the wrong direction on complex tasks.

You either catch it late, or you're constantly babysitting.

## Tweet 3

The solution: Before major decisions, Claude asks Gemini for guidance.

```
oracle ask "Should I use WebSockets or SSE for real-time updates?"
```

Different model, different perspective. Catches issues Claude might miss.

## Tweet 4

But the real magic is `/fullauto` mode.

Tell Claude what you want, walk away, come back to working code.

It autonomously:
- Explores your codebase
- Gets an implementation plan from Gemini
- Executes it
- Validates the result

## Tweet 5

There's also `oracle ask --files` for code review:

```
oracle ask --files src/auth.py "Security issues?"
```

And `oracle imagine` for image generation that auto-provisions a US server if you're geo-restricted (~$0.01/image).

## Tweet 6

The funny part: this repo was created using itself.

I ran `/fullauto make this into a GitHub repo` and it orchestrated the whole thing - structure, install script, README.

The Oracle bootstrapping its own existence.

## Tweet 7

It's open source: github.com/n1ira/claude-oracle

Takes 2 minutes to install. Just need a Gemini API key (free tier works).

Would love feedback from other Claude Code users.

---

**Notes for posting:**
- Include a screenshot or GIF of the CLI in action for Tweet 1
- Best times: Tue-Thu, 9-11am EST
- Tag @AnthropicAI @GoogleAI if feeling bold
- Hashtags to consider: #AI #DevTools #ClaudeCode (don't overdo it)

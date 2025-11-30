# Reddit Post Drafts

---

## r/ClaudeAI

**Title:** Made a CLI that lets Claude Code use Gemini as a "lead architect"

**Body:**

I've been using Claude Code heavily and noticed it sometimes goes off track on complex tasks. Great at coding, but architecture decisions can be hit or miss.

So I built a simple CLI wrapper around Gemini that integrates with Claude Code. The idea is Claude handles the implementation while Gemini provides strategic oversight.

**How it works:**

```bash
# Claude can ask for architectural guidance
oracle ask "What's the best approach for implementing rate limiting?"

# Or get code reviewed
oracle ask --files src/api.py "Any issues with this design?"
```

The interesting part is `/fullauto` mode - you give Claude a task, and it autonomously consults Gemini at key decision points. Like having a pair programming setup where both programmers are AIs.

```
/fullauto implement user authentication with JWT
```

It'll explore your codebase, get a plan from Gemini, execute it, and validate the result. You can walk away and come back to working code.

**Repo:** https://github.com/n1ira/claude-oracle

Install is just `git clone` + `./install.sh`. Need a Gemini API key (free tier works fine).

The meta part: I used this tool to create the repo itself. `/fullauto` orchestrated the whole thing.

Curious if anyone else has experimented with multi-model workflows like this?

---

## r/LocalLLaMA

**Title:** CLI for using Gemini as an orchestrator for Claude Code - includes auto-provisioning for geo-restricted features

**Body:**

Built a tool that pairs Claude Code with Gemini for a "two AI" workflow. Claude writes code, Gemini provides strategic direction.

The part that might interest this sub: **auto-provisioning for geo-restricted image generation**.

Gemini's image gen is blocked in many countries. When you hit the restriction, the CLI automatically:

1. Finds the cheapest US Vast.ai instance (~$0.08/hr)
2. Spins it up
3. Generates the image remotely
4. Downloads it and destroys the instance

Total cost: ~$0.01 per image. Latency is about 60-90 seconds but hey, it works.

```bash
oracle imagine "logo for a terminal app"
# -> detects geo-restriction
# -> provisions us-east instance
# -> generates via API
# -> downloads, destroys instance
```

The main use case is the `/fullauto` mode for Claude Code - high autonomy where Claude consults Gemini at decision points. But the Vast.ai auto-provisioning might be useful for other projects dealing with geo-restrictions.

**Repo:** https://github.com/n1ira/claude-oracle

Code is pretty straightforward if anyone wants to adapt the provisioning logic for other use cases.

---

## r/commandline

**Title:** oracle - CLI for using Gemini as a "second opinion" while coding

**Body:**

Simple CLI that queries Google's Gemini from the terminal. I use it alongside Claude Code for a two-model workflow.

```bash
# Quick questions
oracle quick "best way to parse yaml in python"

# Strategic decisions
oracle ask "microservices vs monolith for a team of 3?"

# Code review with file attachment
oracle ask --files src/db.py "Is this query N+1?"

# Analyze screenshots
oracle ask --image error.png "what's wrong here"

# Image generation
oracle imagine "diagram of request flow"
```

Maintains conversation history per project directory (last 5 exchanges). The `--files` flag supports line ranges like `src/main.py:50-100`.

**Install:**
```bash
git clone https://github.com/n1ira/claude-oracle.git
cd claude-oracle && ./install.sh
export GEMINI_API_KEY="your-key"
```

Written in Python, no dependencies beyond google-genai. MIT license.

**Repo:** https://github.com/n1ira/claude-oracle

---

**Notes for posting:**
- r/ClaudeAI: Focus on the Claude Code integration
- r/LocalLLaMA: Emphasize the Vast.ai auto-provisioning hack
- r/commandline: Keep it tool-focused, less about AI workflows
- Best times: Weekday mornings US time
- Reply to every comment, especially critical ones

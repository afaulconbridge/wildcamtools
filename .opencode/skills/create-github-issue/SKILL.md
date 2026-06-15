---
name: create-github-issue
description: Create a GitHub issue for a code quality problem with full details including problem description, why it matters, and proposed solution
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: github
---
## What I do

This skill creates GitHub issues for code quality problems discovered during code review. Each issue includes:

- Clear, descriptive title
- Detailed problem description
- Explanation of why this matters (security, performance, maintainability, etc.)
- File location with line numbers
- Proposed solution with code examples
- Severity level (critical, high, medium, or low)
- "automated" label to indicate AI-generated issue

## When to use me

Use this skill when:
- A code review has identified a specific issue that needs to be tracked
- You want to create an actionable GitHub issue with full context
- The issue should be labeled as part of an automated review process

## How to use me

Call this skill with the following parameters:

```
skill({ name: "create-github-issue" })
```

Then provide:
- **title**: Clear, concise issue title (e.g., "Security: SQL injection in user auth")
- **file**: File path where the issue exists (e.g., "src/auth.py")
- **line**: Line number(s) where the issue occurs
- **problem**: Detailed description of the problem
- **why_it_matters**: Explanation of impact (security risk, performance, etc.)
- **solution**: Proposed fix with code examples where applicable
- **severity**: One of: critical, high, medium, low

## Example usage

```
skill({ name: "create-github-issue" })

Please create a GitHub issue with:
- title: "Security: Hardcoded API key in configuration"
- file: "src/config.py"
- line: 45
- problem: "API key is hardcoded in the source code instead of using environment variables"
- why_it_matters: "Exposes credentials if code is leaked, violates security best practices"
- solution: "Use os.environ.get('API_KEY') or a secrets manager"
- severity: "critical"
```

## Implementation

The skill will:
1. Ensure the "automated" label exists (create if needed)
2. Create the issue using `gh issue create`
3. Apply the "automated" label
4. Return the issue URL for confirmation

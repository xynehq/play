#!/bin/bash

# Simple PR Analysis - just the 5 commands we need
# Usage: ./pr_analyzer.sh <PR_NUMBER>

if [ -z "$1" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    exit 1
fi

PR_NUMBER=$1

echo "🔍 Step 1: Find PR commit"
GREP_RESULT=$(git log --oneline --all | grep "$PR_NUMBER")

if [ -z "$GREP_RESULT" ]; then
    echo "⚠️  No commit found with PR number in message. Fetching merge commit from GitHub..."
    
    # Set GitHub repo first
    gh repo set-default juspay/hyperswitch 2>/dev/null
    
    # Get merge commit from GitHub
    MERGE_INFO=$(gh pr view $PR_NUMBER --json state,mergedAt,mergeCommit 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        STATE=$(echo "$MERGE_INFO" | jq -r '.state')
        
        if [ "$STATE" = "MERGED" ]; then
            COMMIT_HASH=$(echo "$MERGE_INFO" | jq -r '.mergeCommit.oid' | cut -c1-9)
            echo "✅ Found merge commit: $COMMIT_HASH"
            
            # Verify commit exists locally
            if git cat-file -e $COMMIT_HASH 2>/dev/null; then
                git log --oneline -1 $COMMIT_HASH
            else
                echo "⚠️  Commit not found locally. Fetching from upstream..."
                
                # Check if upstream remote exists
                if git remote | grep -q "^upstream$"; then
                    git fetch upstream
                else
                    echo "Adding upstream remote..."
                    git remote add upstream https://github.com/juspay/hyperswitch.git
                    git fetch upstream
                fi
                
                # Verify again
                if git cat-file -e $COMMIT_HASH 2>/dev/null; then
                    git log --oneline -1 $COMMIT_HASH
                else
                    echo "❌ Still couldn't find commit. Something's wrong."
                    exit 1
                fi
            fi
        else
            echo "❌ PR is not merged yet (state: $STATE)"
            exit 1
        fi
    else
        echo "❌ Could not fetch PR info from GitHub"
        exit 1
    fi
else
    echo "$GREP_RESULT"
    echo ""
    echo "🔍 Step 2: Enter the commit hash you want (e.g., d6e71b959):"
    read COMMIT_HASH
fi

echo ""
echo "🔍 Step 3: Create test branch"
git checkout -b test-claude-pr-$PR_NUMBER ${COMMIT_HASH}^1

echo ""
echo "🔍 Step 4: Set GitHub repo"
gh repo set-default juspay/hyperswitch

echo ""
echo "🔍 Step 5: Get PR info"
PR_INFO=$(gh pr view $PR_NUMBER --json title,body)
echo "$PR_INFO"

echo ""
echo "🔍 Step 6: Get issue info (check PR description for issue number)"
echo "Enter issue number (e.g., 606):"
read ISSUE_NUMBER
ISSUE_INFO=$(gh issue view $ISSUE_NUMBER)
echo "$ISSUE_INFO"

echo ""
echo "🔍 Creating task file..."
TITLE=$(echo "$PR_INFO" | jq -r '.title')
BODY=$(echo "$PR_INFO" | jq -r '.body')

cat > task_pr_${PR_NUMBER}.md << EOF
# Task: $TITLE

## PR Information
**Title:** $TITLE

## Description
$BODY

## Issue #$ISSUE_NUMBER Context
$ISSUE_INFO

## Requirements
Based on the PR description and linked issue above, please implement the required changes.

**Please implement this feature.**
EOF

echo "✅ Done!"
echo "📄 Task saved to: task_pr_${PR_NUMBER}.md"
echo ""
echo "📋 After Claude implements, run these commands:"
echo "=============================================="
echo "# Save Claude's staged changes:"
echo "git diff --cached ${COMMIT_HASH}^1 > claude_changes.diff"
echo ""
echo "# Save original human changes:"
echo "git diff ${COMMIT_HASH}^1 ${COMMIT_HASH} > original_changes.diff"
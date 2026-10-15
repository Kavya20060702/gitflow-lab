# Gitflow Demo

A small Python project for practising Gitflow and GitHub Actions.

## Run locally

```bash
python app.py
python -m pip install pytest ruff
python -m ruff check .
python -m pytest -v
```

## Gitflow practice

Start from the project folder:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit"
git branch develop
git switch develop
git switch -c feature/add-message
```

Make a small change to `app.py`, then:

```bash
git add app.py
git commit -m "Add feature change"
git switch develop
git merge --no-ff feature/add-message -m "Merge feature/add-message"
```

Create an empty GitHub repository, then connect and push (replace USERNAME):

```bash
git remote add origin https://github.com/USERNAME/gitflow-demo.git
git push -u origin main
git push -u origin develop
git push -u origin feature/add-message
```

The GitHub Actions workflow runs Ruff linting and Pytest on pushes and pull requests.

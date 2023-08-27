# Instructions to Add/Edit Pages

1. Create new .rst files for new pages in the source/ directory.
2. Update the index.rst to include these new files in the toctree.
3. To edit existing pages, simply edit the corresponding .rst files.

# Keeping the Website Updated with GitHub Actions

Here's a sample GitHub Actions YAML file to automate the documentation building and deployment process:

```yaml
name: Build and Deploy Docs

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'

    - name: Install Dependencies
      run: pip install sphinx sphinx_rtd_theme

    - name: Build Docs
      run: make html
      working-directory: ./docs

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

After creating this YAML file, push it to the .github/workflows/ directory of your GitHub repository. GitHub Actions will automatically detect this workflow and run it whenever there's a push to the main branch.

Now, you should have a complete system to maintain and update your documentation, along with CI/CD to ensure it stays updated on GitHub Pages.

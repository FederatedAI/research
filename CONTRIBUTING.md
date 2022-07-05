# Contributing

## Welcome
This guide provides guidelines for contributors on how to make new changes into this repository. **Please leave comments / suggestions if you find something is missing or incorrect.**

## Overall Workflow

1. Prepare a contribution proposal document and discuss with maintainers about your contribution.
2. After getting approval from maintainers, make changes to your own fork.
3. PR to `research` **main** branch.

### Contribution Proposal
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the maintainers of this repository before making a change. This is referred as a proposal, which should contain some basic description of your contribution.

### Fork, Clone and Branch
Changes should be made on your own fork in a new branch. To start contributing, fork this repository and clone it to your local workspace. For more details, please refer to the github [official document](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

### Making and Committing Changes
New contributions should follow the file and folder organizing convention in the [README](README.md). New index items should be added in that README file. If new subdirectory needs to be created under one of the top folders, a separate README.md file should be created containing basic description and usage of this contribution.

As this project has integrated the [DCO (Developer Certificate of Origin)](https://probot.github.io/apps/dco/) check tool, contributors are required to sign-off that they adhere to those requirements by adding a Signed-off-by line to the commit messages. Git has even provided a -s command line option to append that automatically to your commit messages, please use it when you commit your changes.
```
$ git commit -s -m 'This is my commit message'
```
The commit message should follow the convention on how to write a git commit message. Be sure to include any related GitHub issue references in the commit message. 


### License
This repository is applying Apache License. If your new contribution adds new subdirectory under the top folders, please include a License and Disclaimer section in your subdirectory README.md file as below:
```
## License and Disclaimer
By downloading or using the work in this folder, you accept and agree to be bound by all of the terms and conditions of the [LICENSE](https://github.com/FederatedAI/research/blob/master/LICENSE) and [DISCLAIMER](https://github.com/FederatedAI/research/blob/master/DISCLAIMER).
```

### Push and Create PR
When ready for review, push your branch to your fork repository on github.com.
Then visit your fork at github.com and click the `Compare & Pull Request` button next to your branch to create a new pull request (PR). The description of PR should contain basic information in the proposal about the contribution.

Once your pull request has been opened it will be assigned to one or more reviewers. Those reviewers will do a thorough code review, looking for correctness, bugs, opportunities for improvement, documentation and comments, and style.

If there are review requiring further changes to the PR, you make new commits into the same branch and push again.

## Advocate
We welcome any articles or blog posts or other forms to advocate the contributions in this repository. If you can let us know the links to the articles or blog posts, we can gather them to the Wiki and help promote them to the community. Additionally, if needed, maintainers in this project may contact and work with you to further promote your contributions.
#!/bin/sh -e

# Ensure black is installed
if ! hash black ; then
  printf "\e[31mblack is not installed.\e[0m
Install black and update your paths.
"
  exit 1
fi

GIT_WORK_TREE="$(git rev-parse --show-toplevel)"
POSTCOMMIT=${GIT_WORK_TREE}/.git/hooks/post-commit

# Ensure a post-commit hook exists
if [ ! -f "${POSTCOMMIT}" ]; then
  printf "\e[33mCreating post-commit hook at ${POSTCOMMIT}.\e[0m\n"
  echo "#!/bin/sh" > "${POSTCOMMIT}"
  chmod a+x "${POSTCOMMIT}"
fi

# Append black formatting command to post-commit hook if it's not already there
if ! grep 'black' ${POSTCOMMIT} >/dev/null ; then
  printf "\e[33mAppending black format command to ${POSTCOMMIT}\e[0m\n"
  cat >> "${POSTCOMMIT}" << 'EOF'
GBF="$(git rev-parse --show-toplevel)/post_commit.git_black_format"
bash ${GBF}
EOF
fi

printf "\e[0;32mPost-commit hook successfully installed for ${GIT_WORK_TREE}\e[0m\n"
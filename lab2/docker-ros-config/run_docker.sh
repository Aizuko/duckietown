if ! [[ -d dotfiles ]]; then
    git clone 'https://gitlab.com/aizuko/dotfiles' dotfiles
fi

docker compose up -d
docker attach $(docker compose ps -q)

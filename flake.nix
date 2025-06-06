{
  description = "Dev flake converted from the old devenv.nix";

  inputs = {
    nixpkgs     .url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils .url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        buildInputs = with pkgs; [
          cudaPackages.cuda_cudart
          cudaPackages.cudatoolkit
          # cudaPackages.cudnn                 # uncomment when you need it
          stdenv.cc.cc
          stdenv.cc.cc.lib
          libuv
          zlib
          ffmpeg
        ];

        devTools = with pkgs; [
          cudaPackages.cuda_nvcc
          git
          bash
          toybox
          uv
        ];

        python = pkgs.python310.withPackages (ps: with ps; [
        ]);

        start = pkgs.writeScriptBin "start" ''
          #!/usr/bin/env sh
          echo 'Running Parakeet STT server'
          python parakeet.py --server --port 5001
        '';

        libPath = pkgs.lib.makeLibraryPath buildInputs;

        env = {
          LD_LIBRARY_PATH =
            "${libPath}:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
          XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}";
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
        };

        exportEnv = pkgs.lib.concatStringsSep "\n" (
          map (n: "export ${n}=\"${env.${n}}\"") (builtins.attrNames env)
        );
      in {
        devShells.default = pkgs.mkShell {
          inherit buildInputs;
          packages = devTools ++ [ python start ];

          # Provide run-time libs automatically
          LD_LIBRARY_PATH = env.LD_LIBRARY_PATH;

          shellHook = ''
            ${exportEnv}

            # optional lightweight local venv – remove if not needed
            if [ ! -d ".venv" ]; then
              echo "Creating local python venv (.venv)… using uv"
              uv venv --python ${python.interpreter}
            fi

            # keep venv in-sync with pyproject/uv.lock
            uv sync                          #
            source .venv/bin/activate

          '';
        };
        apps.parakeet = { type = "app"; program = "${start}/bin/start"; };
      });
}

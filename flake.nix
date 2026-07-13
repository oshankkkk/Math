{
  description = "Python audio processing environment with librosa and LSP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f {
            pkgs = import nixpkgs {
              inherit system;
            };
          });
    in
    {
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: with ps; [
          go
          gopls
		  delve
              librosa
              numpy
              scipy
              matplotlib
              soundfile
              scikit-learn
              numba
			  ipython
			  jupyterlab
			  jupyterlab-vim
              python-lsp-server
            ]))
          ];

          shellHook = ''
            echo "Python librosa environment loaded"
            python --version
      export JUPYTER_APP_DIR="${pkgs.python3.pkgs.jupyterlab}/share/jupyter/lab"
      alias jlab="jupyter lab --app-dir=$JUPYTER_APP_DIR"
          '';
        };
      });
    };
}

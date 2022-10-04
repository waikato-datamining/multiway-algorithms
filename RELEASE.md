How to make a release
=====================

* Run the following command to deploy the artifact:

  ```bash
  mvn release:clean release:prepare release:perform
  ```

* Push all changes
* Update Maven artifact version in [docs/index.md](docs/index.md)

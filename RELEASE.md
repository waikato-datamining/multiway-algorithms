# How to make a release

* Run the following command to deploy the artifact:

  ```
  mvn release:clean release:prepare release:perform
  ```


* Push all changes

* Go to the following URL and publish the artifact:

  ```
  https://central.sonatype.com/publishing/deployments
  ```

* Update Maven artifact version in [docs/index.md](docs/index.md)

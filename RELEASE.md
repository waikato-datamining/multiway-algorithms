How to make a release
=====================

* Run the following command to deploy the artifact:

  ```bash
  mvn release:clean release:prepare release:perform
  ```

* Push all changes
* Push documentation to GitHub Pages: 

  ```bash
  mkdocs gh-deploy
  ```
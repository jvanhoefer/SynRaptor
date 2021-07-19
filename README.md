# SynRaptor

SynRaptor is an open source library for analysing drug synergy screening data. 

### TODOs:
* Features
* How to use/install
* how to get started... 
 

 
### Testing:

This package is developed via test driven development. This means, that tests were written 
before (and independent of) the implementation. After implementing new features you can run
tests by:

* Via the Shell go to the folder of this Package ('SynRaptor')
* Run `pytest test/<your test file>` for running tests
* E.g. `pytest test/test_drugs.py` for all tests related to the drugs file.

Since the tests are implemented and will run even though functionality is not implemented
yet, tests will fail (due to `NotImplementedError`s). This is fine (other reasons to failing
will be due to implementation errors). Over time number of failed tests will (hopefully) approach 0.
 


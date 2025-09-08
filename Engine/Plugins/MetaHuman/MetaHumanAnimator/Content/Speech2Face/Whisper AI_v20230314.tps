<?xml version="1.0" encoding="utf-8"?>
<TpsData xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Name>Whisper AI</Name>
  <!-- Software Name and Version  -->
<!-- Software Name: Whisper AI
    Version: v20230314 -->
<!-- Notes:
 -->
  <Location>Engine\Restricted\NotForLicensees\Plugins\MetaHuman\Content\Speech2Face</Location>
  <Function>Whisper is a pre-trained speech recognition model. It takes audio as input and encodes it as a set of feature vectors. It then decodes these feature vectors into a sequence of words. We only use the audio encoder part of the model - we don't retain the word decoder part when we export it to ONNX format. Internally, the audio is converted to feature vectors through a series of neural network layers based on a Transformer architecture. </Function>
  <Eula>https://github.com/openai/whisper/blob/main/LICENSE</Eula>
  <RedistributeTo>
    <EndUserGroup></EndUserGroup>
    <EndUserGroup>P4</EndUserGroup>
     <EndUserGroup>Git</EndUserGroup>
  </RedistributeTo>
  <LicenseFolder>/Engine/Source/ThirdParty/Licenses</LicenseFolder>
</TpsData>
 
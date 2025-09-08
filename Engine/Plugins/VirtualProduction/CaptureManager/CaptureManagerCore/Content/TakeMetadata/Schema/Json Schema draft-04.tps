<?xml version="1.0" encoding="utf-8"?>
<TpsData xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Name>Json Schema</Name>
  <!-- Software Name and Version  -->
<!-- Software Name: Json Schema
    Download Link: https://json-schema.org/draft-04/schema / https://github.com/json-schema-org/json-schema-spec/tree/draft-04
    Version: draft-04
    Notes: We use RapidJSON (already approved TPS) to validate our input data against our rule set that was created using the template vocabulary as referenced from JSON-Schema.
        -->
<Location>\Engine\Plugins\VirtualProduction\CaptureManager\CaptureManagerCore\Content\TakeMetadata\Schema\</Location>
<Function>We have used their template vocabulary to produce our own rule set used in our code to perform input data validation. There is no use of their code in our software, we have essentially used this as a reference to create our own template.</Function>
<Eula>https://github.com/json-schema-org/json-schema-spec/blob/main/LICENSE</Eula>
  <RedistributeTo>
    <EndUserGroup>Licensee</EndUserGroup>
    <EndUserGroup>P4</EndUserGroup>
    <EndUserGroup>Git</EndUserGroup>
  </RedistributeTo>
  <LicenseFolder>/Engine/Source/ThirdParty/Licenses</LicenseFolder>
</TpsData>
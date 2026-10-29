param(
    [string[]]$WorkbookPaths
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression.FileSystem
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Get-WorksheetTargetMap {
    param(
        [xml]$WorkbookXml,
        [xml]$WorkbookRelsXml
    )

    $mainNs = New-Object System.Xml.XmlNamespaceManager($WorkbookXml.NameTable)
    $mainNs.AddNamespace("d", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    $mainNs.AddNamespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")

    $relsNs = New-Object System.Xml.XmlNamespaceManager($WorkbookRelsXml.NameTable)
    $relsNs.AddNamespace("d", "http://schemas.openxmlformats.org/package/2006/relationships")

    $relMap = @{}
    foreach ($rel in $WorkbookRelsXml.SelectNodes("//d:Relationship", $relsNs)) {
        $target = $rel.Target.TrimStart("/")
        if ($target -notmatch "^xl/") {
            $target = "xl/" + $target
        }
        $relMap[$rel.Id] = $target
    }

    $sheetMap = @()
    foreach ($sheet in $WorkbookXml.SelectNodes("//d:sheets/d:sheet", $mainNs)) {
        $sheetMap += [pscustomobject]@{
            Name = $sheet.name
            SheetId = [int]$sheet.sheetId
            RelId = $sheet.GetAttribute("id", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
            Target = $relMap[$sheet.GetAttribute("id", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")]
        }
    }

    return $sheetMap
}

function Get-CellValue {
    param(
        [System.Xml.XmlElement]$Cell,
        [string[]]$SharedStrings
    )

    if ($null -eq $Cell) {
        return $null
    }

    $cellType = $Cell.GetAttribute("t")
    $valueNode = $Cell.SelectSingleNode("./*[local-name()='v']")
    $inlineTextNode = $Cell.SelectSingleNode("./*[local-name()='is']/*[local-name()='t']")

    if ($cellType -eq "inlineStr") {
        if ($null -ne $inlineTextNode) {
            return [string]$inlineTextNode.InnerText
        }
        return ""
    }

    if ($cellType -eq "s" -and $null -ne $valueNode) {
        $index = [int]$valueNode.InnerText
        if ($index -ge 0 -and $index -lt $SharedStrings.Count) {
            return $SharedStrings[$index]
        }
    }

    if ($null -ne $valueNode) {
        return [string]$valueNode.InnerText
    }

    return $null
}

function Get-SheetAverages {
    param(
        [string]$XmlContent,
        [string[]]$MetricHeaders,
        [string[]]$SharedStrings
    )

    $sheetXml = New-Object xml
    $sheetXml.LoadXml($XmlContent)

    $ns = New-Object System.Xml.XmlNamespaceManager($sheetXml.NameTable)
    $ns.AddNamespace("d", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")

    $rows = $sheetXml.SelectNodes("//d:sheetData/d:row", $ns)
    if ($rows.Count -lt 2) {
        throw "Worksheet has no data rows."
    }

    $headerMap = @{}
    foreach ($cell in $rows[0].SelectNodes("./d:c", $ns)) {
        $ref = [string]$cell.r
        $columnRef = ($ref -replace "\d", "")
            $value = Get-CellValue -Cell $cell -SharedStrings $SharedStrings
        if ($value) {
            $headerMap[$value] = $columnRef
        }
    }

    $averages = [ordered]@{}
    foreach ($metric in $MetricHeaders) {
        if (-not $headerMap.ContainsKey($metric)) {
            throw "Metric header '$metric' not found."
        }

        $sum = 0.0
        $count = 0
        $targetColumn = $headerMap[$metric]

        for ($i = 1; $i -lt $rows.Count; $i++) {
            $row = $rows[$i]
            $cell = $row.SelectSingleNode("./d:c[starts-with(@r, '$targetColumn')]", $ns)
            $rawValue = Get-CellValue -Cell $cell -SharedStrings $SharedStrings
            if ($null -ne $rawValue -and $rawValue -ne "") {
                $sum += [double]::Parse($rawValue, [System.Globalization.CultureInfo]::InvariantCulture)
                $count++
            }
        }

        if ($count -eq 0) {
            $averages[$metric] = ""
        } else {
            $averages[$metric] = ($sum / $count).ToString("0.###############", [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }

    return $averages
}

function New-InlineStringCellXml {
    param(
        [string]$CellRef,
        [string]$Value
    )

    $escaped = [System.Security.SecurityElement]::Escape($Value)
    return "<c r=`"$CellRef`" t=`"inlineStr`"><is><t>$escaped</t></is></c>"
}

function New-NumberCellXml {
    param(
        [string]$CellRef,
        [string]$Value
    )

    return "<c r=`"$CellRef`"><v>$Value</v></c>"
}

function New-AvgSheetXml {
    param(
        [object[]]$Rows
    )

    $headers = @(
        "x",
        "bleu",
        "rouge_1_f1",
        "rouge_2_f1",
        "rouge_l_f1",
        "meteor",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "f1",
        "precision",
        "accuracy",
        "recall"
    )

    $columns = @("A","B","C","D","E","F","G","H","I","J","K","L","M")
    $rowXml = New-Object System.Collections.Generic.List[string]

    $headerCells = for ($i = 0; $i -lt $headers.Count; $i++) {
        New-InlineStringCellXml -CellRef ($columns[$i] + "1") -Value $headers[$i]
    }
    $rowXml.Add("<row r=`"1`">" + ($headerCells -join "") + "</row>")

    for ($rowIndex = 0; $rowIndex -lt $Rows.Count; $rowIndex++) {
        $excelRow = $rowIndex + 2
        $item = $Rows[$rowIndex]
        $cells = New-Object System.Collections.Generic.List[string]
        $cells.Add((New-NumberCellXml -CellRef ("A" + $excelRow) -Value ([string]$item.x)))
        $cells.Add((New-NumberCellXml -CellRef ("B" + $excelRow) -Value $item.bleu))
        $cells.Add((New-NumberCellXml -CellRef ("C" + $excelRow) -Value $item.rouge_1_f1))
        $cells.Add((New-NumberCellXml -CellRef ("D" + $excelRow) -Value $item.rouge_2_f1))
        $cells.Add((New-NumberCellXml -CellRef ("E" + $excelRow) -Value $item.rouge_l_f1))
        $cells.Add((New-NumberCellXml -CellRef ("F" + $excelRow) -Value $item.meteor))
        $cells.Add((New-NumberCellXml -CellRef ("G" + $excelRow) -Value $item.bertscore_p))
        $cells.Add((New-NumberCellXml -CellRef ("H" + $excelRow) -Value $item.bertscore_r))
        $cells.Add((New-NumberCellXml -CellRef ("I" + $excelRow) -Value $item.bertscore_f1))
        $cells.Add((New-NumberCellXml -CellRef ("J" + $excelRow) -Value $item.f1))
        $cells.Add((New-NumberCellXml -CellRef ("K" + $excelRow) -Value $item.precision))
        $cells.Add((New-NumberCellXml -CellRef ("L" + $excelRow) -Value $item.accuracy))
        $cells.Add((New-NumberCellXml -CellRef ("M" + $excelRow) -Value $item.recall))
        $rowXml.Add("<row r=`"$excelRow`">" + ($cells -join "") + "</row>")
    }

    return @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:M$($Rows.Count + 1)"/>
  <sheetViews>
    <sheetView workbookViewId="0"/>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <sheetData>
    $($rowXml -join "`n    ")
  </sheetData>
  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>
</worksheet>
"@
}

function Update-XmlDeclaration {
    param(
        [xml]$XmlDocument
    )

    $xmlText = $XmlDocument.OuterXml -replace '^\s*(<\?xml[^>]+\?>\s*)+', ''
    return "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" + $xmlText
}

function Read-XmlDocument {
    param(
        [string]$Path
    )

    $raw = Get-Content -LiteralPath $Path -Raw
    $normalized = $raw -replace '^\s*(<\?xml[^>]+\?>\s*)+', ''
    return [xml]$normalized
}

function Get-SharedStrings {
    param(
        [string]$ExtractDir
    )

    $sharedStringsPath = Join-Path $ExtractDir "xl\sharedStrings.xml"
    if (-not (Test-Path -LiteralPath $sharedStringsPath)) {
        return @()
    }

    [xml]$sharedStringsXml = Get-Content -LiteralPath $sharedStringsPath -Raw
    $ns = New-Object System.Xml.XmlNamespaceManager($sharedStringsXml.NameTable)
    $ns.AddNamespace("d", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")

    $values = New-Object System.Collections.Generic.List[string]
    foreach ($si in $sharedStringsXml.SelectNodes("//d:sst/d:si", $ns)) {
        $textNodes = $si.SelectNodes(".//d:t", $ns)
        if ($textNodes.Count -eq 0) {
            $values.Add("")
            continue
        }
        $parts = foreach ($node in $textNodes) { [string]$node.InnerText }
        $values.Add(($parts -join ""))
    }
    return $values.ToArray()
}

function New-ZipFromDirectory {
    param(
        [string]$SourceDirectory,
        [string]$DestinationZip
    )

    $fileStream = [System.IO.File]::Open($DestinationZip, [System.IO.FileMode]::Create)
    try {
        $zipArchive = New-Object System.IO.Compression.ZipArchive($fileStream, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            $files = Get-ChildItem -LiteralPath $SourceDirectory -Recurse -File
            foreach ($file in $files) {
                $relativePath = $file.FullName.Substring($SourceDirectory.Length).TrimStart("\")
                $entryName = $relativePath -replace "\\", "/"
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zipArchive, $file.FullName, $entryName, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
            }
        }
        finally {
            $zipArchive.Dispose()
        }
    }
    finally {
        $fileStream.Dispose()
    }
}

if (-not $WorkbookPaths -or $WorkbookPaths.Count -eq 0) {
    throw "Provide at least one workbook path."
}

$metricHeaders = @(
    "bleu",
    "rouge_1_f1",
    "rouge_2_f1",
    "rouge_l_f1",
    "meteor",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "f1",
    "precision",
    "accuracy",
    "recall"
)

$baseTempDir = Join-Path $env:TEMP ("avg_score_sheet_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $baseTempDir | Out-Null

try {
    foreach ($workbookPath in $WorkbookPaths) {
        $extractDir = Join-Path $baseTempDir ([IO.Path]::GetFileNameWithoutExtension($workbookPath))
        [System.IO.Compression.ZipFile]::ExtractToDirectory($workbookPath, $extractDir)

        $workbookXmlPath = Join-Path $extractDir "xl\workbook.xml"
        $workbookRelsPath = Join-Path $extractDir "xl\_rels\workbook.xml.rels"
        $contentTypesPath = Join-Path $extractDir "[Content_Types].xml"
        $appXmlPath = Join-Path $extractDir "docProps\app.xml"

        [xml]$workbookXml = Read-XmlDocument -Path $workbookXmlPath
        [xml]$workbookRelsXml = Read-XmlDocument -Path $workbookRelsPath
        [xml]$contentTypesXml = Read-XmlDocument -Path $contentTypesPath
        [xml]$appXml = Read-XmlDocument -Path $appXmlPath
        $sharedStrings = Get-SharedStrings -ExtractDir $extractDir

        $sheetMap = Get-WorksheetTargetMap -WorkbookXml $workbookXml -WorkbookRelsXml $workbookRelsXml
        $avgRows = New-Object System.Collections.Generic.List[object]

        foreach ($sheetInfo in ($sheetMap | Where-Object { $_.Name -like "x=*" } | Sort-Object SheetId)) {
            $sheetPath = Join-Path $extractDir $sheetInfo.Target.Replace("/", "\")
            $sheetXmlText = Get-Content -LiteralPath $sheetPath -Raw
            $averages = Get-SheetAverages -XmlContent $sheetXmlText -MetricHeaders $metricHeaders -SharedStrings $sharedStrings
            $avgRows.Add([pscustomobject]@{
                x = [int](($sheetInfo.Name -replace "^x=", ""))
                bleu = $averages["bleu"]
                rouge_1_f1 = $averages["rouge_1_f1"]
                rouge_2_f1 = $averages["rouge_2_f1"]
                rouge_l_f1 = $averages["rouge_l_f1"]
                meteor = $averages["meteor"]
                bertscore_p = $averages["bertscore_p"]
                bertscore_r = $averages["bertscore_r"]
                bertscore_f1 = $averages["bertscore_f1"]
                f1 = $averages["f1"]
                precision = $averages["precision"]
                accuracy = $averages["accuracy"]
                recall = $averages["recall"]
            })
        }

        $existingAvg = $sheetMap | Where-Object { $_.Name -eq "avg score" }
        if ($existingAvg) {
            $existingSheetPath = Join-Path $extractDir $existingAvg.Target.Replace("/", "\")
            $avgSheetXml = New-AvgSheetXml -Rows ($avgRows | Sort-Object x)
            [System.IO.File]::WriteAllText($existingSheetPath, $avgSheetXml, $Utf8NoBom)

            [System.IO.File]::WriteAllText($workbookXmlPath, (Update-XmlDeclaration -XmlDocument $workbookXml), $Utf8NoBom)
            [System.IO.File]::WriteAllText($workbookRelsPath, (Update-XmlDeclaration -XmlDocument $workbookRelsXml), $Utf8NoBom)
            [System.IO.File]::WriteAllText($contentTypesPath, (Update-XmlDeclaration -XmlDocument $contentTypesXml), $Utf8NoBom)
            [System.IO.File]::WriteAllText($appXmlPath, (Update-XmlDeclaration -XmlDocument $appXml), $Utf8NoBom)

            $rebuiltZip = Join-Path $baseTempDir ([IO.Path]::GetFileNameWithoutExtension($workbookPath) + "_rebuilt.zip")
            if (Test-Path -LiteralPath $rebuiltZip) {
                Remove-Item -LiteralPath $rebuiltZip -Force
            }
            New-ZipFromDirectory -SourceDirectory $extractDir -DestinationZip $rebuiltZip
            Move-Item -LiteralPath $rebuiltZip -Destination $workbookPath -Force

            Write-Output "Updated: $workbookPath"
            continue
        }

        $newSheetId = (($sheetMap | Measure-Object -Property SheetId -Maximum).Maximum) + 1
        $worksheetDir = Join-Path $extractDir "xl\worksheets"
        $existingSheetNumbers = Get-ChildItem -LiteralPath $worksheetDir -Filter "sheet*.xml" |
            ForEach-Object {
                if ($_.BaseName -match "^sheet(\d+)$") { [int]$matches[1] }
            }
        $newSheetNumber = (($existingSheetNumbers | Measure-Object -Maximum).Maximum) + 1
        $newSheetFileName = "sheet$newSheetNumber.xml"
        $newSheetRelativePath = "xl/worksheets/$newSheetFileName"
        $newSheetPath = Join-Path $worksheetDir $newSheetFileName

        $avgSheetXml = New-AvgSheetXml -Rows ($avgRows | Sort-Object x)
        [System.IO.File]::WriteAllText($newSheetPath, $avgSheetXml, $Utf8NoBom)

        $mainNs = New-Object System.Xml.XmlNamespaceManager($workbookXml.NameTable)
        $mainNs.AddNamespace("d", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
        $sheetsNode = $workbookXml.SelectSingleNode("//d:sheets", $mainNs)
        $newSheetNode = $workbookXml.CreateElement("sheet", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
        $null = $newSheetNode.SetAttribute("name", "avg score")
        $null = $newSheetNode.SetAttribute("sheetId", [string]$newSheetId)
        $null = $newSheetNode.SetAttribute("state", "visible")
        [void]$sheetsNode.AppendChild($newSheetNode)

        $relsNs = New-Object System.Xml.XmlNamespaceManager($workbookRelsXml.NameTable)
        $relsNs.AddNamespace("d", "http://schemas.openxmlformats.org/package/2006/relationships")
        $relationshipsNode = $workbookRelsXml.SelectSingleNode("//d:Relationships", $relsNs)
        $existingRelIds = $workbookRelsXml.SelectNodes("//d:Relationship", $relsNs) | ForEach-Object {
            if ($_.Id -match "^rId(\d+)$") { [int]$matches[1] }
        }
        $newRelNumber = (($existingRelIds | Measure-Object -Maximum).Maximum) + 1
        $newRelationship = $workbookRelsXml.CreateElement("Relationship", "http://schemas.openxmlformats.org/package/2006/relationships")
        $null = $newRelationship.SetAttribute("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet")
        $null = $newRelationship.SetAttribute("Target", "worksheets/$newSheetFileName")
        $null = $newRelationship.SetAttribute("Id", "rId$newRelNumber")
        [void]$relationshipsNode.AppendChild($newRelationship)
        $null = $newSheetNode.SetAttribute("id", "http://schemas.openxmlformats.org/officeDocument/2006/relationships", "rId$newRelNumber")

        $contentNs = New-Object System.Xml.XmlNamespaceManager($contentTypesXml.NameTable)
        $contentNs.AddNamespace("d", "http://schemas.openxmlformats.org/package/2006/content-types")
        $typesNode = $contentTypesXml.SelectSingleNode("//d:Types", $contentNs)
        $overrideNode = $contentTypesXml.CreateElement("Override", "http://schemas.openxmlformats.org/package/2006/content-types")
        $null = $overrideNode.SetAttribute("PartName", "/$newSheetRelativePath")
        $null = $overrideNode.SetAttribute("ContentType", "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
        [void]$typesNode.AppendChild($overrideNode)

        $appNs = New-Object System.Xml.XmlNamespaceManager($appXml.NameTable)
        $appNs.AddNamespace("ep", "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties")
        $appNs.AddNamespace("vt", "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes")
        $titlesVector = $appXml.SelectSingleNode("//ep:TitlesOfParts/vt:vector", $appNs)
        $worksheetsNode = $appXml.SelectSingleNode("//ep:HeadingPairs/vt:vector/vt:variant[1]/vt:i4", $appNs)
        if ($null -ne $titlesVector -and $null -ne $worksheetsNode) {
            $newLpstr = $appXml.CreateElement("lpstr", "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes")
            $newLpstr.InnerText = "avg score"
            [void]$titlesVector.AppendChild($newLpstr)
            $null = $titlesVector.SetAttribute("size", [string]([int]$titlesVector.GetAttribute("size") + 1))
            $worksheetsNode.InnerText = [string]([int]$worksheetsNode.InnerText + 1)
        }

        [System.IO.File]::WriteAllText($workbookXmlPath, (Update-XmlDeclaration -XmlDocument $workbookXml), $Utf8NoBom)
        [System.IO.File]::WriteAllText($workbookRelsPath, (Update-XmlDeclaration -XmlDocument $workbookRelsXml), $Utf8NoBom)
        [System.IO.File]::WriteAllText($contentTypesPath, (Update-XmlDeclaration -XmlDocument $contentTypesXml), $Utf8NoBom)
        [System.IO.File]::WriteAllText($appXmlPath, (Update-XmlDeclaration -XmlDocument $appXml), $Utf8NoBom)

        $rebuiltZip = Join-Path $baseTempDir ([IO.Path]::GetFileNameWithoutExtension($workbookPath) + "_rebuilt.zip")
        if (Test-Path -LiteralPath $rebuiltZip) {
            Remove-Item -LiteralPath $rebuiltZip -Force
        }
        New-ZipFromDirectory -SourceDirectory $extractDir -DestinationZip $rebuiltZip
        Move-Item -LiteralPath $rebuiltZip -Destination $workbookPath -Force

        Write-Output "Updated: $workbookPath"
    }
}
finally {
    if (Test-Path -LiteralPath $baseTempDir) {
        Remove-Item -LiteralPath $baseTempDir -Recurse -Force
    }
}
